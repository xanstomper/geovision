"""
GeoVision Deepfake & Manipulation Detector
==========================================
Six forensic techniques fused into authenticity_score + verdict.
Uses only: opencv-python, Pillow, numpy, exifread (all in requirements.txt)

Verdicts: LIKELY_GENUINE | SUSPICIOUS | LIKELY_MANIPULATED | LIKELY_AI_GENERATED
"""
from __future__ import annotations
import base64, io, logging, os, tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_THRESH_GENUINE    = 0.70
_THRESH_SUSPICIOUS = 0.45

_AI_SOFTWARE_TAGS = {
    "stable diffusion","stablediffusion","automatic1111","comfyui",
    "midjourney","dall-e","dall·e","adobe firefly","firefly",
    "bing image creator","image creator","adobe photoshop generative",
    "generative fill","canva ai","nightcafe","runway","leonardo ai",
    "invokeai","fooocus","flux","imagen","sora",
}

_AI_DIMENSIONS = {
    (512,512),(512,768),(768,512),(768,768),(1024,1024),
    (1024,576),(576,1024),(1280,720),(720,1280),(1024,768),(768,1024),
}


class DeepfakeDetector:
    def analyze(self, image_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status":"failed","authenticity_score":0.5,"manipulation_probability":0.5,
            "verdict":"UNKNOWN","confidence":0.0,"techniques":{},"flags":[],"summary":"",
        }
        try:
            pil_img = Image.open(image_path).convert("RGB")
            cv_img  = cv2.imread(str(image_path))
            if cv_img is None:
                result["summary"] = "Could not open image."
                return result

            ela   = self._ela(image_path, pil_img)
            freq  = self._frequency(cv_img)
            meta  = self._metadata(image_path, pil_img)
            noise = self._noise(cv_img)
            face  = self._face(cv_img)
            ghost = self._ghost(image_path, pil_img)

            result["techniques"] = {"ela":ela,"frequency":freq,"metadata":meta,
                                    "noise":noise,"face":face,"compression_ghost":ghost}

            flags = []
            for t in result["techniques"].values():
                flags.extend(t.get("flags",[]))
            result["flags"] = flags

            weights = {"ela":0.25,"frequency":0.20,"metadata":0.30,"noise":0.15,"face":0.05,"compression_ghost":0.05}
            scores  = {"ela":ela.get("authenticity",0.5),"frequency":freq.get("authenticity",0.5),
                       "metadata":meta.get("authenticity",0.5),"noise":noise.get("authenticity",0.5),
                       "face":face.get("authenticity",0.5),"compression_ghost":ghost.get("authenticity",0.5)}

            total_w = fused = 0.0
            for k,w in weights.items():
                s = scores[k]
                if s is None: continue
                fused += w*s; total_w += w
            fused = fused/total_w if total_w else 0.5

            auth = float(np.clip(fused,0.0,1.0))
            result["authenticity_score"]       = round(auth,4)
            result["manipulation_probability"] = round(1.0-auth,4)

            if auth >= _THRESH_GENUINE:                verdict = "LIKELY_GENUINE"
            elif auth >= _THRESH_SUSPICIOUS:           verdict = "SUSPICIOUS"
            elif meta.get("ai_software_detected") or freq.get("has_periodic_pattern"):
                                                       verdict = "LIKELY_AI_GENERATED"
            else:                                      verdict = "LIKELY_MANIPULATED"
            result["verdict"]     = verdict
            result["confidence"]  = round(min(1.0,abs(auth-0.5)*2.0),4)
            result["summary"]     = self._summary(verdict,auth,flags)
            result["status"]      = "success"
        except Exception as e:
            logger.error("DeepfakeDetector: %s",e,exc_info=True)
            result["summary"] = f"Error: {e}"
        return result

    # -- ELA --
    def _ela(self, image_path, pil_img) -> Dict[str,Any]:
        out={"authenticity":0.5,"flags":[],"mean_ela":None,"high_ela_pct":None,"ela_map_base64":None}
        try:
            tmp=tempfile.NamedTemporaryFile(suffix=".jpg",delete=False); tmp.close()
            pil_img.save(tmp.name,"JPEG",quality=90)
            ela_img=Image.open(tmp.name).convert("RGB"); os.unlink(tmp.name)
            orig=np.array(pil_img,dtype=np.float32); comp=np.array(ela_img,dtype=np.float32)
            diff=np.abs(orig-comp)
            mean_ela=float(diff.mean()); std_ela=float(diff.std()); high_pct=float((diff>20).mean())
            out.update({"mean_ela":round(mean_ela,3),"high_ela_pct":round(high_pct,4)})
            # ELA map
            ela_map=(diff*10).clip(0,255).astype(np.uint8)
            small=Image.fromarray(ela_map).resize((200,200))
            buf=io.BytesIO(); small.save(buf,"PNG")
            out["ela_map_base64"]=base64.b64encode(buf.getvalue()).decode()
            if mean_ela<2.0:
                out["authenticity"]=0.35; out["flags"].append("ELA: Unusually low error level — possible AI generation")
            elif high_pct>0.30:
                out["authenticity"]=0.25; out["flags"].append(f"ELA: {high_pct*100:.1f}% pixels have high ELA — possible manipulation")
            elif std_ela<1.0 and mean_ela<8.0:
                out["authenticity"]=0.40; out["flags"].append("ELA: Suspiciously uniform — consistent with AI generation")
            else:
                out["authenticity"]=0.75
        except Exception as e:
            logger.warning("ELA: %s",e)
        return out

    # -- DCT Frequency --
    def _frequency(self, cv_img) -> Dict[str,Any]:
        out={"authenticity":0.5,"flags":[],"has_periodic_pattern":False,"high_freq_ratio":None}
        try:
            gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray=cv2.resize(gray,(512,512))
            fft=np.fft.fft2(gray); mag=np.abs(np.fft.fftshift(fft)); log=np.log1p(mag)
            h,w=log.shape; cy,cx=h//2,w//2
            Y,X=np.ogrid[:h,:w]; dist=np.sqrt((Y-cy)**2+(X-cx)**2)
            low_mask=dist<min(h,w)//8; high_mask=dist>min(h,w)//3
            ratio=float(log[high_mask].mean())/float(log[low_mask].mean()+1e-8)
            out["high_freq_ratio"]=round(ratio,4)
            axis=np.zeros((h,w),bool)
            axis[cy-3:cy+4,:]=True; axis[:,cx-3:cx+4]=True; axis[low_mask]=False
            if axis.any():
                ap=float(log[axis].max()); op=float(log[~axis&(dist>min(h,w)//8)].max())
                periodic=ap>op*1.4; out["has_periodic_pattern"]=periodic
                if periodic:
                    out["authenticity"]=0.30
                    out["flags"].append("Frequency: Periodic grid pattern in FFT — diffusion model artifact")
                else: out["authenticity"]=0.70
            if ratio>0.85:
                out["authenticity"]=min(out["authenticity"],0.40)
                out["flags"].append(f"Frequency: High-frequency energy ratio {ratio:.2f}")
        except Exception as e:
            logger.warning("Freq: %s",e)
        return out

    # -- Metadata --
    def _metadata(self, image_path, pil_img) -> Dict[str,Any]:
        out={"authenticity":0.5,"flags":[],"ai_software_detected":False,"software":None,"has_camera_make":False}
        try:
            exif={}
            try:
                from PIL.ExifTags import TAGS
                raw=pil_img._getexif() or {}
                exif={TAGS.get(k,k):str(v) for k,v in raw.items()}
            except Exception: pass
            try:
                import exifread
                with open(image_path,"rb") as f:
                    tags=exifread.process_file(f,stop_tag="UNDEF",details=False)
                exif.update({k:str(v) for k,v in tags.items()})
            except Exception: pass
            sw=exif.get("Software",exif.get("Image Software",""))
            out["software"]=sw or None
            if sw:
                for tag in _AI_SOFTWARE_TAGS:
                    if tag in sw.lower():
                        out["ai_software_detected"]=True
                        out["flags"].append(f"Metadata: AI software tag — '{sw}'"); break
            has_make=bool(exif.get("Make") or exif.get("Image Make"))
            out["has_camera_make"]=has_make
            if not has_make: out["flags"].append("Metadata: No camera Make/Model in EXIF")
            w,h=pil_img.size
            if (w,h) in _AI_DIMENSIONS or (h,w) in _AI_DIMENSIONS:
                out["flags"].append(f"Metadata: Dimensions {w}×{h} match common AI output size")
            has_dt=bool(exif.get("EXIF DateTimeOriginal") or exif.get("DateTimeOriginal") or exif.get("Image DateTime"))
            if not has_dt and not has_make: out["flags"].append("Metadata: No capture timestamp or camera info")
            if out["ai_software_detected"]: out["authenticity"]=0.05
            elif not has_make and not has_dt: out["authenticity"]=0.25
            elif not has_make: out["authenticity"]=0.40
            else: out["authenticity"]=0.85
        except Exception as e:
            logger.warning("Meta: %s",e)
        return out

    # -- Noise / PRNU --
    def _noise(self, cv_img) -> Dict[str,Any]:
        out={"authenticity":0.5,"flags":[],"noise_uniformity":None}
        try:
            gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY).astype(np.float32)
            blurred=cv2.GaussianBlur(gray,(5,5),0); noise=gray-blurred
            h,w=noise.shape; bh,bw=h//8,w//8
            stds=[float(noise[r*bh:(r+1)*bh,c*bw:(c+1)*bw].std()) for r in range(8) for c in range(8)]
            arr=np.array(stds); uniformity=1.0-min(1.0,arr.std()/(arr.mean()+1e-8))
            out["noise_uniformity"]=round(float(uniformity),4)
            if uniformity>0.90:
                out["authenticity"]=0.25; out["flags"].append(f"Noise: High uniformity ({uniformity:.2f}) — AI generation signature")
            elif uniformity>0.75:
                out["authenticity"]=0.45; out["flags"].append("Noise: Moderately uniform noise pattern")
            else: out["authenticity"]=0.78
        except Exception as e:
            logger.warning("Noise: %s",e)
        return out

    # -- Face Consistency --
    def _face(self, cv_img) -> Dict[str,Any]:
        out={"authenticity":0.5,"flags":[],"face_detected":False,"face_count":0}
        try:
            cp=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
            if not Path(cp).exists(): return out
            gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
            faces=cv2.CascadeClassifier(cp).detectMultiScale(gray,1.1,5,minSize=(60,60))
            if not isinstance(faces,np.ndarray) or len(faces)==0: return out
            out["face_detected"]=True; out["face_count"]=len(faces)
            sus=0.0
            for (fx,fy,fw,fh) in faces:
                crop=gray[fy:fy+fh,fx:fx+fw]
                if crop.size==0: continue
                sym=float(np.abs(crop.astype(float)-cv2.flip(crop,1).astype(float)).mean())
                if sym<5.0: sus+=0.4; out["flags"].append("Face: Unnaturally high bilateral symmetry — GAN artifact")
                edges=cv2.Canny(crop,50,150)
                if float(edges.mean())<2.0: sus+=0.3; out["flags"].append("Face: Very smooth face edges")
            out["authenticity"]=max(0.1,1.0-sus/len(faces))
        except Exception as e:
            logger.warning("Face: %s",e)
        return out

    # -- Compression Ghost --
    def _ghost(self, image_path, pil_img) -> Dict[str,Any]:
        out={"authenticity":0.5,"flags":[],"ghost_variance":None}
        try:
            orig=np.array(pil_img,dtype=np.float32); diffs=[]
            for q in (75,85,95):
                tmp=tempfile.NamedTemporaryFile(suffix=".jpg",delete=False); tmp.close()
                pil_img.save(tmp.name,"JPEG",quality=q)
                diffs.append(float(np.abs(orig-np.array(Image.open(tmp.name).convert("RGB"),dtype=np.float32)).mean()))
                os.unlink(tmp.name)
            var=float(np.var(diffs)); out["ghost_variance"]=round(var,4)
            if var<0.5:
                out["authenticity"]=0.30; out["flags"].append("Compression: Very low ghost variance — no prior JPEG history")
            elif var>10.0: out["authenticity"]=0.80
            else: out["authenticity"]=0.60
        except Exception as e:
            logger.warning("Ghost: %s",e)
        return out

    def _summary(self, verdict, score, flags):
        pct=round(score*100)
        m={"LIKELY_GENUINE":f"Appears to be a genuine photograph (authenticity {pct}%).",
           "SUSPICIOUS":f"Shows suspicious characteristics (authenticity {pct}%). Further review recommended.",
           "LIKELY_MANIPULATED":f"Likely contains digital manipulation (authenticity {pct}%). {len(flags)} indicator(s) found.",
           "LIKELY_AI_GENERATED":f"Likely AI-generated (authenticity {pct}%). Synthetic signatures detected."}
        base=m.get(verdict,f"Authenticity: {pct}%.")
        if flags: base+=f" Key: {'; '.join(flags[:2])}."
        return base


if __name__=="__main__":
    import sys,json
    r=DeepfakeDetector().analyze(sys.argv[1] if len(sys.argv)>1 else "building_image.jpg")
    print(json.dumps({k:v for k,v in r.items() if k!="techniques"},indent=2))
