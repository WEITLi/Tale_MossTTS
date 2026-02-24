import io, os, gc, torch, hashlib, traceback, uvicorn, sys, multiprocessing, time, subprocess
import soundfile as sf
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import queue

# ==========================================
# 0. 系统配置
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

QWEN_LIBS = os.environ.get("QWEN_LIBS", "./qwen_libs")
QWEN_MODEL = "./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# MOSS 模型配置 (侧载目录)
MOSS_LIBS = os.environ.get("MOSS_LIBS", "./moss_libs")
MOSS_VOICE_MODEL = os.environ.get("MOSS_VOICE_MODEL", "./models/MOSS-VoiceGenerator")
MOSS_SFX_MODEL = os.environ.get("MOSS_SFX_MODEL", "./models/MOSS-SoundEffect")
MOSS_DEVICE = os.environ.get("MOSS_DEVICE", "cuda:0")

PROMPTS_DIR = os.path.abspath("prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)

app = FastAPI(title="Super Unitale Smart API")

class ForceCORS(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return Response(status_code=200, headers={
                "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Credentials": "false",
            })
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

app.add_middleware(ForceCORS)

def hash_filename(filename: str) -> str:
    ext = os.path.splitext(filename)[1] or ".wav"
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{h}{ext}"

# ==========================================
# 1. Qwen3 守护进程逻辑
# ==========================================
def qwen_daemon(input_q, output_q):
    model = None
    try:
        sys.path.insert(0, QWEN_LIBS)
        # 将用户的 Qwen3-TTS 库路径加入
        qwen_repo_path = "./Qwen3-TTS"
        if qwen_repo_path not in sys.path:
            sys.path.insert(0, qwen_repo_path)
            
        import torch
        import sox
        from qwen_tts import Qwen3TTSModel
        
        print(f"🟢 [Qwen Daemon] 子进程启动 (PID: {os.getpid()})，正在加载模型...")
        torch.cuda.empty_cache()
        
        model = Qwen3TTSModel.from_pretrained(
            QWEN_MODEL, 
            device_map="cuda:0", 
            dtype=torch.bfloat16, 
            local_files_only=True
        )
        print(f"🟢 [Qwen Daemon] 模型加载完毕，等待指令...")
        
        while True:
            try:
                task = input_q.get(timeout=1) 
            except queue.Empty:
                continue

            if task.get("command") == "STOP":
                print("🔴 [Qwen Daemon] 收到停止指令，正在退出...")
                break
            
            if task.get("command") == "DESIGN":
                try:
                    print(f"🔵 [Qwen Daemon] 开始处理音色合成任务...")
                    req_dict = task["data"]
                    wavs, sr = model.generate_voice_design(
                        text=req_dict.get("text", "预览"),
                        language="Chinese",
                        instruct=req_dict["voice_description"]
                    )
                    
                    audio_data = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else wavs[0]
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, sr, format="WAV")
                    buf.seek(0)
                    
                    output_q.put({"success": True, "audio_bytes": buf.read()})
                    print(f"🔵 [Qwen Daemon] 任务完成")
                except Exception as e:
                    traceback.print_exc()
                    output_q.put({"success": False, "error": str(e)})
                    
    except Exception as e:
        print(f"❌ [Qwen Daemon] 致命错误: {e}")
        traceback.print_exc()
    finally:
        if model: del model
        gc.collect()
        torch.cuda.empty_cache()
        print("🔴 [Qwen Daemon] 进程销毁，显存释放")

# ==========================================
# 1b. MOSS 辅助函数与守护进程
# ==========================================
def resolve_moss_attn(requested, device, dtype):
    """照搬 MOSS 官方 attn 选择逻辑"""
    import importlib.util
    norm = (requested or "").strip().lower()
    if norm == "none": return None
    if norm not in {"", "auto"}: return requested
    if device.type == "cuda" and importlib.util.find_spec("flash_attn") and dtype in {torch.float16, torch.bfloat16}:
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8: return "flash_attention_2"
    return "sdpa" if device.type == "cuda" else "eager"

def moss_voice_daemon(input_q, output_q):
    """MOSS-VoiceGenerator 守护进程 (声音设计)"""
    model = None
    try:
        sys.path.insert(0, MOSS_LIBS)
        import torch, io as _io, numpy as np, soundfile as _sf
        from transformers import AutoModel, AutoProcessor
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        device = torch.device(MOSS_DEVICE if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        attn_impl = resolve_moss_attn("auto", device, dtype)

        print(f"🟢 [MOSS Voice Daemon] 子进程启动 (PID: {os.getpid()})，正在加载模型...")
        torch.cuda.empty_cache()

        processor = AutoProcessor.from_pretrained(MOSS_VOICE_MODEL, trust_remote_code=True, normalize_inputs=True)
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if attn_impl: model_kwargs["attn_implementation"] = attn_impl
        model = AutoModel.from_pretrained(MOSS_VOICE_MODEL, **model_kwargs).to(device)
        model.eval()
        sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
        print(f"🟢 [MOSS Voice Daemon] 模型加载完毕 (SR={sample_rate})，等待指令...")

        while True:
            try:
                task = input_q.get(timeout=1)
            except queue.Empty:
                continue
            if task.get("command") == "STOP":
                print("🔴 [MOSS Voice Daemon] 收到停止指令，正在退出...")
                break
            if task.get("command") == "DESIGN":
                try:
                    print(f"🔵 [MOSS Voice Daemon] 开始处理音色合成任务...")
                    req = task["data"]
                    conversations = [[processor.build_user_message(
                        text=req.get("text", "预览"), instruction=req["voice_description"]
                    )]]
                    batch = processor(conversations, mode="generation")
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            max_new_tokens=4096, audio_temperature=1.5,
                            audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1,
                        )
                    messages = processor.decode(outputs)
                    audio = messages[0].audio_codes_list[0]
                    audio_np = audio.detach().float().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, dtype=np.float32)
                    if audio_np.ndim > 1: audio_np = audio_np.reshape(-1)
                    buf = _io.BytesIO()
                    _sf.write(buf, audio_np.astype(np.float32), sample_rate, format="WAV")
                    buf.seek(0)
                    output_q.put({"success": True, "audio_bytes": buf.read()})
                    print(f"🔵 [MOSS Voice Daemon] 任务完成")
                except Exception as e:
                    traceback.print_exc()
                    output_q.put({"success": False, "error": str(e)})
    except Exception as e:
        print(f"❌ [MOSS Voice Daemon] 致命错误: {e}")
        traceback.print_exc()
    finally:
        if model: del model
        gc.collect()
        torch.cuda.empty_cache()
        print("🔴 [MOSS Voice Daemon] 进程销毁，显存释放")

def moss_sfx_daemon(input_q, output_q):
    """MOSS-SoundEffect 守护进程 (音效生成)"""
    model = None
    try:
        sys.path.insert(0, MOSS_LIBS)
        import torch, io as _io, numpy as np, soundfile as _sf
        from transformers import AutoModel, AutoProcessor
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        device = torch.device(MOSS_DEVICE if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        attn_impl = resolve_moss_attn("auto", device, dtype)

        print(f"🟢 [MOSS SFX Daemon] 子进程启动 (PID: {os.getpid()})，正在加载模型...")
        torch.cuda.empty_cache()

        processor = AutoProcessor.from_pretrained(MOSS_SFX_MODEL, trust_remote_code=True)
        if hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if attn_impl: model_kwargs["attn_implementation"] = attn_impl
        model = AutoModel.from_pretrained(MOSS_SFX_MODEL, **model_kwargs).to(device)
        model.eval()
        sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
        TOKENS_PER_SECOND = 12.5
        print(f"🟢 [MOSS SFX Daemon] 模型加载完毕 (SR={sample_rate})，等待指令...")

        while True:
            try:
                task = input_q.get(timeout=1)
            except queue.Empty:
                continue
            if task.get("command") == "STOP":
                print("🔴 [MOSS SFX Daemon] 收到停止指令，正在退出...")
                break
            if task.get("command") == "GENERATE_SFX":
                try:
                    print(f"🔵 [MOSS SFX Daemon] 开始处理音效生成任务...")
                    req = task["data"]
                    duration = float(req.get("duration_seconds", 5))
                    expected_tokens = max(1, int(duration * TOKENS_PER_SECOND))
                    conversations = [[processor.build_user_message(
                        ambient_sound=req["ambient_sound"], tokens=expected_tokens
                    )]]
                    batch = processor(conversations, mode="generation")
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            max_new_tokens=4096, audio_temperature=1.5,
                            audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.2,
                        )
                    messages = processor.decode(outputs)
                    audio = messages[0].audio_codes_list[0]
                    audio_np = audio.detach().float().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio, dtype=np.float32)
                    if audio_np.ndim > 1: audio_np = audio_np.reshape(-1)
                    buf = _io.BytesIO()
                    _sf.write(buf, audio_np.astype(np.float32), sample_rate, format="WAV")
                    buf.seek(0)
                    output_q.put({"success": True, "audio_bytes": buf.read()})
                    print(f"🔵 [MOSS SFX Daemon] 任务完成")
                except Exception as e:
                    traceback.print_exc()
                    output_q.put({"success": False, "error": str(e)})
    except Exception as e:
        print(f"❌ [MOSS SFX Daemon] 致命错误: {e}")
        traceback.print_exc()
    finally:
        if model: del model
        gc.collect()
        torch.cuda.empty_cache()
        print("🔴 [MOSS SFX Daemon] 进程销毁，显存释放")

# ==========================================
# 2. 全局模型管理器 (精准白名单版)
# ==========================================
class ModelManager:
    def __init__(self):
        self.indextts = None
        # Qwen3 进程 (保留)
        self.qwen_process: Optional[multiprocessing.Process] = None
        self.qwen_in_q: Optional[multiprocessing.Queue] = None
        self.qwen_out_q: Optional[multiprocessing.Queue] = None
        # MOSS Voice 进程 (新增)
        self.moss_voice_process: Optional[multiprocessing.Process] = None
        self.moss_voice_in_q: Optional[multiprocessing.Queue] = None
        self.moss_voice_out_q: Optional[multiprocessing.Queue] = None
        # MOSS SFX 进程 (新增)
        self.moss_sfx_process: Optional[multiprocessing.Process] = None
        self.moss_sfx_in_q: Optional[multiprocessing.Queue] = None
        self.moss_sfx_out_q: Optional[multiprocessing.Queue] = None
        
        self.lock = multiprocessing.Lock()
        
        # 记录主进程 PID
        self.main_pid = os.getpid()
        
        if multiprocessing.current_process().name == 'MainProcess':
            self._init_resident_model()

    def _init_resident_model(self):
        print(f"[启动] 主进程 PID: {self.main_pid}")
        print("[启动] 正在载入 IndexTTS2...")
        
        # 将用户的 index-tts 路径加入 sys.path
        indextts_repo_path = "./index-tts"
        if indextts_repo_path not in sys.path:
            sys.path.insert(0, indextts_repo_path)
            
        from indextts.infer_v2 import IndexTTS2
        
        indextts_model_dir = "./models/IndexTTS-2"
        self.indextts = IndexTTS2(
            model_dir=indextts_model_dir,
            cfg_path=os.path.join(indextts_model_dir, "config.yaml")
        )
        print("✅ IndexTTS2 就绪。")

    def _kill_zombies(self):
        """[清道夫] 最终精细版：建立白名单，不再误伤管家和自己"""
        try:
            cmd = "ps -eo pid,args | grep python"
            output = subprocess.check_output(cmd, shell=True, encoding='utf-8')
            
            for line in output.strip().split('\n'):
                if not line: continue
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2: continue
                
                try:
                    pid = int(parts[0])
                    cmdline = parts[1]
                except ValueError:
                    continue

                # 🛑 豁免名单 (Whitelist)
                if pid == self.main_pid: continue
                if self.qwen_process and self.qwen_process.is_alive() and pid == self.qwen_process.pid: continue
                if self.moss_voice_process and self.moss_voice_process.is_alive() and pid == self.moss_voice_process.pid: continue
                if self.moss_sfx_process and self.moss_sfx_process.is_alive() and pid == self.moss_sfx_process.pid: continue
                if "resource_tracker" in cmdline: continue
                if "grep" in cmdline or "ps -eo" in cmdline: continue

                print(f"💀 [内部清洗] 发现未知 Python 进程 PID: {pid} ({cmdline[:15]}...)，执行清理...")
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
        except Exception:
            pass

    # --- Qwen3 进程管理 (保留) ---
    def ensure_qwen_loaded(self):
        if self.qwen_process is not None and self.qwen_process.is_alive():
            return 

        print("🟡 [调度器] 准备启动 Qwen3...")
        self._kill_zombies() 
        
        print("🧹 [调度器] 正在压缩 IndexTTS 显存碎片...")
        torch.cuda.empty_cache()
        gc.collect()

        print("🟡 [调度器] 拉起 Qwen3 守护进程...")
        self.qwen_in_q = multiprocessing.Queue()
        self.qwen_out_q = multiprocessing.Queue()
        self.qwen_process = multiprocessing.Process(
            target=qwen_daemon, 
            args=(self.qwen_in_q, self.qwen_out_q),
            daemon=True
        )
        self.qwen_process.start()

    def unload_qwen(self):
        if self.qwen_process is not None and self.qwen_process.is_alive():
            print("⚠️ [调度器] 正在卸载 Qwen3...")
            try:
                self.qwen_in_q.put({"command": "STOP"})
                self.qwen_process.join(timeout=3)
                if self.qwen_process.is_alive():
                    self.qwen_process.terminate()
                    self.qwen_process.join()
            except Exception:
                pass
            
            self.qwen_process = None
            self.qwen_in_q = None
            self.qwen_out_q = None
            torch.cuda.empty_cache()
            print("✅ [调度器] Qwen3 已卸载")

    # --- MOSS Voice 进程管理 (新增) ---
    def ensure_moss_voice_loaded(self):
        if self.moss_voice_process is not None and self.moss_voice_process.is_alive():
            return

        print("🟡 [调度器] 准备启动 MOSS VoiceGenerator...")
        self._kill_zombies()
        torch.cuda.empty_cache()
        gc.collect()

        self.moss_voice_in_q = multiprocessing.Queue()
        self.moss_voice_out_q = multiprocessing.Queue()
        self.moss_voice_process = multiprocessing.Process(
            target=moss_voice_daemon,
            args=(self.moss_voice_in_q, self.moss_voice_out_q),
            daemon=True
        )
        self.moss_voice_process.start()

    def unload_moss_voice(self):
        if self.moss_voice_process is not None and self.moss_voice_process.is_alive():
            print("⚠️ [调度器] 正在卸载 MOSS Voice...")
            try:
                self.moss_voice_in_q.put({"command": "STOP"})
                self.moss_voice_process.join(timeout=3)
                if self.moss_voice_process.is_alive():
                    self.moss_voice_process.terminate()
                    self.moss_voice_process.join()
            except Exception:
                pass
            self.moss_voice_process = None
            self.moss_voice_in_q = None
            self.moss_voice_out_q = None
            torch.cuda.empty_cache()
            print("✅ [调度器] MOSS Voice 已卸载")

    # --- MOSS SFX 进程管理 (新增) ---
    def ensure_moss_sfx_loaded(self):
        if self.moss_sfx_process is not None and self.moss_sfx_process.is_alive():
            return

        print("🟡 [调度器] 准备启动 MOSS SoundEffect...")
        self._kill_zombies()
        torch.cuda.empty_cache()
        gc.collect()

        self.moss_sfx_in_q = multiprocessing.Queue()
        self.moss_sfx_out_q = multiprocessing.Queue()
        self.moss_sfx_process = multiprocessing.Process(
            target=moss_sfx_daemon,
            args=(self.moss_sfx_in_q, self.moss_sfx_out_q),
            daemon=True
        )
        self.moss_sfx_process.start()

    def unload_moss_sfx(self):
        if self.moss_sfx_process is not None and self.moss_sfx_process.is_alive():
            print("⚠️ [调度器] 正在卸载 MOSS SFX...")
            try:
                self.moss_sfx_in_q.put({"command": "STOP"})
                self.moss_sfx_process.join(timeout=3)
                if self.moss_sfx_process.is_alive():
                    self.moss_sfx_process.terminate()
                    self.moss_sfx_process.join()
            except Exception:
                pass
            self.moss_sfx_process = None
            self.moss_sfx_in_q = None
            self.moss_sfx_out_q = None
            torch.cuda.empty_cache()
            print("✅ [调度器] MOSS SFX 已卸载")

manager = ModelManager()

# ==========================================
# 3. 接口定义
# ==========================================
class TextToSpeechRequest(BaseModel):
    text: str 
    audio_path: str 
    emo_text: Optional[str] = None
    emo_vector: Optional[List[float]] = Field(None, min_items=8, max_items=8)

class QwenDesignRequest(BaseModel):
    voice_description: str
    text: str = "这是生成的参考音频预览。"
    save_as: Optional[str] = "designed_voice.wav" 

@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    content = await audio.read()
    save_path = os.path.join(PROMPTS_DIR, hash_filename(full_path))
    with open(save_path, "wb") as f: f.write(content)
    return {"code": 200, "msg": "上传成功", "filename": full_path}

@app.get("/v1/check/audio")
async def check_audio_exists(file_name: str):
    exists = os.path.isfile(os.path.join(PROMPTS_DIR, hash_filename(file_name)))
    return {"code": 200 if exists else 404, "exists": exists}

@app.post("/v1/qwen/design")
async def qwen_design(request: QwenDesignRequest):
    with manager.lock:
        manager.ensure_qwen_loaded()
        manager.qwen_in_q.put({"command": "DESIGN", "data": request.model_dump()})
        try:
            res = manager.qwen_out_q.get(timeout=120) 
            if res.get("success"):
                return Response(content=res["audio_bytes"], media_type="audio/wav")
            else:
                raise HTTPException(status_code=500, detail=res.get("error"))
        except queue.Empty:
            manager.unload_qwen()
            raise HTTPException(status_code=500, detail="Qwen 推理超时")

class MossSfxRequest(BaseModel):
    ambient_sound: str
    duration_seconds: float = 5.0
    save_as: Optional[str] = None

@app.post("/v1/moss/design")
async def moss_design(request: QwenDesignRequest):
    """使用 MOSS-VoiceGenerator 生成声音（与 /v1/qwen/design 接口格式一致）"""
    with manager.lock:
        manager.ensure_moss_voice_loaded()
        manager.moss_voice_in_q.put({"command": "DESIGN", "data": request.model_dump()})
        try:
            res = manager.moss_voice_out_q.get(timeout=300)
            if res.get("success"):
                return Response(content=res["audio_bytes"], media_type="audio/wav")
            else:
                raise HTTPException(status_code=500, detail=res.get("error"))
        except queue.Empty:
            manager.unload_moss_voice()
            raise HTTPException(status_code=500, detail="MOSS 推理超时")

@app.post("/v1/moss/sfx")
async def moss_sfx_generate(request: MossSfxRequest):
    """使用 MOSS-SoundEffect 从文字描述生成音效"""
    with manager.lock:
        manager.ensure_moss_sfx_loaded()
        manager.moss_sfx_in_q.put({"command": "GENERATE_SFX", "data": request.model_dump()})
        try:
            res = manager.moss_sfx_out_q.get(timeout=300)
            if res.get("success"):
                return Response(content=res["audio_bytes"], media_type="audio/wav")
            else:
                raise HTTPException(status_code=500, detail=res.get("error"))
        except queue.Empty:
            manager.unload_moss_sfx()
            raise HTTPException(status_code=500, detail="MOSS SFX 推理超时")

@app.post("/v2/synthesize")
async def synthesize_v2(request: TextToSpeechRequest):
    manager.unload_qwen()
    manager.unload_moss_voice()  # 释放 MOSS 显存
    manager.unload_moss_sfx()    # 释放 MOSS SFX 显存
    manager._kill_zombies() # 双重保险
    
    if manager.indextts is None: raise HTTPException(status_code=503, detail="IndexTTS2 未就绪")
    
    real_file_path = os.path.join(PROMPTS_DIR, hash_filename(request.audio_path))
    temp_out = os.path.join(PROMPTS_DIR, f"temp_synth_{time.time()}.wav")
    if not os.path.isfile(real_file_path): raise HTTPException(status_code=404, detail="音频不存在")

    try:
        manager.indextts.infer(
            spk_audio_prompt=real_file_path, text=request.text, output_path=temp_out,
            emo_vector=request.emo_vector, emo_text=request.emo_text,
            use_emo_text=bool(request.emo_text), emo_alpha=0.6
        )
        with open(temp_out, "rb") as f: data = f.read()
        if os.path.exists(temp_out): os.remove(temp_out)
        return Response(content=data, media_type="audio/wav")
    except Exception as e:
        if os.path.exists(temp_out): os.remove(temp_out)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    uvicorn.run(app, host="0.0.0.0", port=8300)