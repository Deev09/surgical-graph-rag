"""MVP-v1 — self-contained interactive 3D viewer over the frozen results.

  python3 tools/mvp_viewer.py [--out runs/mvp_v1/viewer.html]
                              [--scenes replica_office_0 replica_room_2]

Spec: docs/mvp_v1_viewer_spec.md (owner-specified). Emits ONE offline HTML
file: full-resolution colored point clouds of the raw mesh.ply vertices,
rendered by a hand-written WebGL renderer (no three.js, no CDN, zero
external requests), with per-vertex oracle ids (from the semantic mesh)
and C1 instance ids (from the frozen ms02 bundle) for overlay toggles,
click-to-inspect, and question-driven highlighting.

Every answer, citation, status, and verbalization comes VERBATIM from the
MVP-v0 deterministic reports (runs/mvp_v0/<scene>_mvp.json — run
tools/mvp_demo.py first); the viewer computes no metrics. Deterministic:
byte-identical output for identical inputs.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_habitat_import import _gravity_align_matrix, _aligned_structural_surfaces
from demo.replica_mesh_import import _parse_semantic_ply
from segmenter.base import load_segmentation_output
from segmenter.derived import ROOM_0_Z_TRANSLATION
from segmenter.ply import parse_vertices_with_colors
from tools.c1_exact_eval import evaluate, oracle_vertex_membership
from tools.mvp_report_html import DISCLOSURES

DEFAULT_SCENES = ("replica_office_0", "replica_room_2")
MS02_DIR = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02"
MANIFEST = REPO_ROOT / "docs" / "c1_artifact_manifest.json"
SIDECAR_DIR = REPO_ROOT / "eval" / "predictions" / "phase8_c2"
SCENE_MANIFEST = REPO_ROOT / "eval" / "questions" / "phase8" / "scene_manifest.json"
MVP_DIR = REPO_ROOT / "runs" / "mvp_v0"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(a.tobytes()).decode("ascii")


def build_scene_payload(room_dir: Path, bundle_dir: Path, scene_id: str,
                        mvp_report: dict) -> dict:
    """All per-scene viewer data. Positions are uint16-quantized over the
    bbox in the SAME gravity-aligned frame the pipeline uses."""
    xyz, rgb = parse_vertices_with_colors(room_dir / "mesh.ply")
    info = json.loads((room_dir / "habitat" / "info_semantic.json").read_text())
    g = info["gravity_dir"]
    R0 = _gravity_align_matrix((float(g[0]), float(g[1]), float(g[2])))
    R, _, _, _ = _aligned_structural_surfaces(info, R0, ROOM_0_Z_TRANSLATION)
    xyz = np.einsum("ij,nj->ni", R, xyz)

    n = len(xyz)
    seg = load_segmentation_output(bundle_dir)
    if seg.n_vertices != n:
        raise ValueError(f"{scene_id}: bundle/mesh vertex count mismatch")
    pred = seg.vertex_instance_ids.astype(np.int16)
    _, vidx, oid = _parse_semantic_ply(room_dir / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, n).astype(np.int16)

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    q = np.round((xyz - lo) / span * 65535.0).astype(np.uint16)

    ev = evaluate(room_dir, bundle_dir)
    matches = {m["oracle_id"]: m for m in ev["matches"]}
    pred_match = {m["pred_id"]: m for m in ev["matches"]}
    c2 = {}
    sc_path = SIDECAR_DIR / f"{scene_id}_c2_labels.json"
    if sc_path.exists():
        sidecar = json.loads(sc_path.read_text())
        if sidecar["applies_to_bundle_output_sha256"] != seg.output_sha256:
            raise ValueError(f"{scene_id}: C2 sidecar pinned to another bundle")
        c2 = {r["pred_id"]: r["learned_label"]
              for r in sidecar["per_instance"]}

    objects = {}
    for o in info["objects"]:
        i = int(o["id"])
        m = matches.get(i)
        objects[str(i)] = {
            "label": o.get("class_name", "?"),
            "pred": m["pred_id"] if m else None,
            "iou": round(m["iou"], 3) if m else None,
            "c2": c2.get(m["pred_id"]) if m else None,
        }
    preds = {}
    for p in sorted(set(int(x) for x in np.unique(pred) if x >= 0)):
        m = pred_match.get(p)
        preds[str(p)] = {
            "oracle": m["oracle_id"] if m else None,
            "iou": round(m["iou"], 3) if m else None,
            "c2": c2.get(p),
        }

    slim_variants = {}
    for v, r in mvp_report["variants"].items():
        slim_variants[v] = {k: r.get(k) for k in (
            "micro_precision", "micro_recall", "semantic_citation",
            "n_graph_edges", "entity_matches_at_05", "questions")}
    return {
        "scene_id": scene_id,
        "n": n,
        "bbox": [[round(float(x), 6) for x in lo],
                 [round(float(x), 6) for x in hi]],
        "b64": {"pos": _b64(q), "rgb": _b64(np.ascontiguousarray(rgb)),
                "oracle": _b64(oracle), "pred": _b64(pred)},
        "objects": objects,
        "preds": preds,
        "key_questions": mvp_report["key_questions"],
        "variants": slim_variants,
        "c1_status": mvp_report["c1_status"],
        "provenance": {
            "git_commit": mvp_report["provenance"]["git_commit"],
            "key": mvp_report["provenance"]["key"],
            "c1_bundle": mvp_report["provenance"].get("c1_bundle"),
            "isolation": mvp_report["provenance"]["isolation_statement"],
        },
    }


def build_viewer_html(payloads: list[dict]) -> str:
    data = {
        "scenes": {p["scene_id"]: p for p in payloads},
        "scene_order": [p["scene_id"] for p in payloads],
        "disclosures": [{"title": t, "body": b} for t, b in DISCLOSURES],
        "evaluation_only": ("C1 rows inject oracle labels+surfaces "
                            "(instance boundaries learned); C2 rows use "
                            "frozen zero-shot labels from the committed "
                            "sidecars. Both are EVALUATION-ONLY ladder "
                            "stages, not deployable raw-scene QA."),
    }
    blob = json.dumps(data, sort_keys=True).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", blob)


_TEMPLATE = r"""<!doctype html><meta charset="utf-8">
<title>MVP-v1 — 3D evidence viewer</title>
<style>
html,body{margin:0;height:100%;font:13px/1.45 -apple-system,'Segoe UI',sans-serif;background:#111;color:#ddd;overflow:hidden}
#side{position:absolute;left:0;top:0;bottom:0;width:360px;overflow-y:auto;background:#1a1a1c;border-right:1px solid #333;padding:12px;box-sizing:border-box}
#gl{position:absolute;left:360px;top:0;right:0;bottom:0}
canvas{display:block;width:100%;height:100%}
h1{font-size:15px;margin:0 0 8px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#888;margin:14px 0 6px}
.btnrow{display:flex;gap:4px;flex-wrap:wrap}
button{background:#2a2a2e;color:#ddd;border:1px solid #444;border-radius:5px;padding:4px 10px;cursor:pointer;font-size:12px}
button.on{background:#3a6ea5;border-color:#5b8fc7;color:#fff}
select{width:100%;background:#2a2a2e;color:#ddd;border:1px solid #444;border-radius:5px;padding:4px}
.badge{display:inline-block;border-radius:4px;padding:1px 7px;font-size:11px;font-weight:600;margin-left:6px}
.b-answer{background:#1a7a3a;color:#fff}.b-empty{background:#555;color:#fff}
.b-defer{background:#8a6d00;color:#fff}.b-unknown{background:#6a3fa0;color:#fff}
.hit{color:#2ecc71}.wrong{color:#e74c3c}.missed{color:#f39c12}.anchor{color:#3498db}
.small{color:#888;font-size:11px}
.panel{background:#222226;border:1px solid #333;border-radius:6px;padding:8px;margin:6px 0}
.disc{background:#26221a;border:1px solid #4a4022;border-radius:6px;padding:7px;margin:5px 0;font-size:11px;color:#cbb}
.legend span{margin-right:10px}
kbd{background:#333;border-radius:3px;padding:0 4px}
</style>
<div id="side">
 <h1>MVP-v1 — 3D evidence viewer</h1>
 <div class="small">Drag = orbit · wheel = zoom · shift-drag = pan · click = inspect object</div>
 <h2>Scene</h2><div class="btnrow" id="sceneBtns"></div>
 <h2>Overlay</h2><div class="btnrow" id="modeBtns"></div>
 <h2>Answer source (3D highlight)</h2><div class="btnrow" id="variantBtns"></div>
 <div class="small" id="variantStats"></div>
 <h2>Question</h2>
 <select id="qSel"><option value="">— none (free look) —</option></select>
 <div id="answerPanel"></div>
 <h2>Clicked object</h2><div id="pickPanel" class="panel small">nothing selected</div>
 <h2>Evaluation-only notice</h2><div class="disc" id="evalNote"></div>
 <h2>Disclosures</h2><div id="discs"></div>
 <h2>Provenance</h2><div id="prov" class="panel small"></div>
</div>
<div id="gl"><canvas id="c"></canvas></div>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl", {antialias:false, preserveDrawingBuffer:false});
if (!gl) document.body.textContent = "WebGL unavailable";

// ---------- tiny mat4 ----------
function persp(f, asp, near, far){const t=1/Math.tan(f/2);return [t/asp,0,0,0, 0,t,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0];}
function lookAt(e,c,u){let z=norm3(sub3(e,c)),x=norm3(cross3(u,z)),y=cross3(z,x);
 return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot3(x,e),-dot3(y,e),-dot3(z,e),1];}
function mul4(a,b){const o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
function sub3(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross3(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot3(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function norm3(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];}

// ---------- shaders ----------
function shader(type, src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
 if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(s);return s;}
const VS=`attribute vec3 aPos;attribute vec3 aCol;uniform mat4 uMVP;uniform float uPt;
varying vec3 vCol;void main(){gl_Position=uMVP*vec4(aPos,1.0);
gl_PointSize=clamp(uPt/gl_Position.w,1.0,7.0);vCol=aCol;}`;
const FS=`precision mediump float;varying vec3 vCol;void main(){gl_FragColor=vec4(vCol,1.0);}`;
const prog=gl.createProgram();
gl.attachShader(prog,shader(gl.VERTEX_SHADER,VS));
gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);gl.useProgram(prog);
const locPos=gl.getAttribLocation(prog,"aPos"),locCol=gl.getAttribLocation(prog,"aCol");
const locMVP=gl.getUniformLocation(prog,"uMVP"),locPt=gl.getUniformLocation(prog,"uPt");

// ---------- state ----------
let S=null;                       // active scene record (decoded)
const cache={};
let mode="rgb", variant="A", qid="", cam=null;

function b64bytes(s){const bin=atob(s);const a=new Uint8Array(bin.length);
 for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i);return a;}

function decodeScene(sid){
 if(cache[sid])return cache[sid];
 const p=DATA.scenes[sid];
 const q=new Uint16Array(b64bytes(p.b64.pos).buffer);
 const rgb=b64bytes(p.b64.rgb);
 const oracle=new Int16Array(b64bytes(p.b64.oracle).buffer);
 const pred=new Int16Array(b64bytes(p.b64.pred).buffer);
 const n=p.n, lo=p.bbox[0], hi=p.bbox[1];
 const pos=new Float32Array(3*n);
 for(let a=0;a<3;a++){const s=(hi[a]-lo[a])/65535, o=lo[a];
  for(let i=0;i<n;i++)pos[3*i+a]=o+q[3*i+a]*s;}
 const idCol=new Uint8Array(3*n);
 for(let i=0;i<n;i++){idCol[3*i]=i&255;idCol[3*i+1]=(i>>8)&255;idCol[3*i+2]=(i>>16)&255;}
 const rec={p,n,pos,rgb,oracle,pred,idCol,
  posBuf:gl.createBuffer(),colBuf:gl.createBuffer(),idBuf:gl.createBuffer(),
  disp:new Uint8Array(3*n)};
 gl.bindBuffer(gl.ARRAY_BUFFER,rec.posBuf);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
 gl.bindBuffer(gl.ARRAY_BUFFER,rec.idBuf);gl.bufferData(gl.ARRAY_BUFFER,idCol,gl.STATIC_DRAW);
 cache[sid]=rec;return rec;
}

function hashColor(id){let h=(id*2654435761)>>>0;
 return [60+(h&127), 60+((h>>7)&127), 60+((h>>14)&127)];}
const C_HIT=[46,204,113],C_WRONG=[231,76,60],C_MISS=[243,156,18],C_ANCHOR=[52,152,219];

function activeQuestionRecord(){
 if(!qid)return null;
 const v=S.p.variants[variant==="Human"?firstVariant():variant];
 if(!v||!v.questions)return null;
 return v.questions.find(x=>x.question_id===qid)||null;
}
function firstVariant(){return Object.keys(S.p.variants)[0];}

function classMaps(){
 // per-oracle-id and per-pred-id highlight classes: 1 hit 2 wrong 3 missed 4 anchor
 const oidC={}, pidC={};
 if(!qid)return {oidC,pidC};
 const kq=S.p.key_questions[qid];
 if(variant==="Human"){
  for(const u of kq.expected_must_contain)oidC[u.replace("obj_","")]=1;
  for(const u of kq.expected_must_not_contain)oidC[u.replace("obj_","")]=2;
 } else {
  const qr=activeQuestionRecordFor(variant);
  if(qr){
   for(const c of qr.cited){
    const cls=c.status==="hit"?1:2;
    if(c.uid.startsWith("pred:")) pidC[c.uid.replace("pred:obj_","")]=cls;
    else if(variant==="C1"||variant==="C2"){
     const o=c.uid.replace("obj_","");
     const pr=(S.p.objects[o]||{}).pred;
     if(pr!==null&&pr!==undefined)pidC[pr]=cls; else oidC[o]=cls;
    } else oidC[c.uid.replace("obj_","")]=cls;
   }
   for(const u of qr.missed)oidC[u.replace("obj_","")]=3;
  }
 }
 // anchor emphasis for support questions, under the ACTIVE variant's labels
 const m=(kq.question||"").match(/on the ([a-z\-]+)\?/);
 if(m&&kq.relation==="ON_ENTITY_SURFACE"){
  const cls=m[1];
  if(variant==="C2"){
   for(const[pid,inf]of Object.entries(S.p.preds))
    if(inf.c2===cls&&pidC[pid]===undefined)pidC[pid]=4;
  } else if(variant==="C1"){
   for(const[o,inf]of Object.entries(S.p.objects))
    if(inf.label===cls&&inf.pred!==null&&pidC[inf.pred]===undefined)pidC[inf.pred]=4;
  } else {
   for(const[o,inf]of Object.entries(S.p.objects))
    if(inf.label===cls&&oidC[o]===undefined)oidC[o]=4;
  }
 }
 return {oidC,pidC};
}
function activeQuestionRecordFor(v){
 const vv=S.p.variants[v];
 if(!vv||!vv.questions)return null;
 return vv.questions.find(x=>x.question_id===qid)||null;
}

function recolor(){
 const {oidC,pidC}=classMaps();
 const n=S.n,d=S.disp,rgb=S.rgb,orc=S.oracle,prd=S.pred;
 const dim=qid?0.28:1.0;
 for(let i=0;i<n;i++){
  let r,g,b;
  if(mode==="rgb"){r=rgb[3*i];g=rgb[3*i+1];b=rgb[3*i+2];}
  else if(mode==="oracle"){const o=orc[i];if(o<0){r=g=b=55;}else{const c=hashColor(o);r=c[0];g=c[1];b=c[2];}}
  else{const p2=prd[i];if(p2<0){r=g=b=55;}else{const c=hashColor(p2);r=c[0];g=c[1];b=c[2];}}
  let cls;
  const o=orc[i],p2=prd[i];
  cls=(o>=0?oidC[o]:undefined);
  if(cls===undefined&&p2>=0)cls=pidC[p2];
  if(cls===1){r=C_HIT[0];g=C_HIT[1];b=C_HIT[2];}
  else if(cls===2){r=C_WRONG[0];g=C_WRONG[1];b=C_WRONG[2];}
  else if(cls===3){r=C_MISS[0];g=C_MISS[1];b=C_MISS[2];}
  else if(cls===4){r=C_ANCHOR[0];g=C_ANCHOR[1];b=C_ANCHOR[2];}
  else{r*=dim;g*=dim;b*=dim;}
  d[3*i]=r;d[3*i+1]=g;d[3*i+2]=b;
 }
 gl.bindBuffer(gl.ARRAY_BUFFER,S.colBuf);
 gl.bufferData(gl.ARRAY_BUFFER,d,gl.DYNAMIC_DRAW);
 draw();
}

// ---------- camera / draw ----------
function resetCam(){
 const lo=S.p.bbox[0],hi=S.p.bbox[1];
 const c=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
 const r=Math.hypot(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
 cam={center:c,theta:-1.2,phi:1.0,radius:r*0.9,pan:[0,0,0]};
}
function draw(){
 const w=canvas.clientWidth,h=canvas.clientHeight;
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 gl.viewport(0,0,w,h);
 gl.clearColor(0.07,0.07,0.08,1);gl.enable(gl.DEPTH_TEST);
 gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 if(!S)return;
 const eye=eyePos();
 const ctr=[cam.center[0]+cam.pan[0],cam.center[1]+cam.pan[1],cam.center[2]+cam.pan[2]];
 const mvp=mul4(persp(0.9,w/h,0.05,200),lookAt(eye,ctr,[0,0,1]));
 gl.uniformMatrix4fv(locMVP,false,new Float32Array(mvp));
 gl.uniform1f(locPt,h*0.011);
 gl.bindBuffer(gl.ARRAY_BUFFER,S.posBuf);
 gl.enableVertexAttribArray(locPos);gl.vertexAttribPointer(locPos,3,gl.FLOAT,false,0,0);
 gl.bindBuffer(gl.ARRAY_BUFFER,S.colBuf);
 gl.enableVertexAttribArray(locCol);gl.vertexAttribPointer(locCol,3,gl.UNSIGNED_BYTE,true,0,0);
 gl.drawArrays(gl.POINTS,0,S.n);
}
function eyePos(){
 const ctr=[cam.center[0]+cam.pan[0],cam.center[1]+cam.pan[1],cam.center[2]+cam.pan[2]];
 return [ctr[0]+cam.radius*Math.cos(cam.phi)*Math.cos(cam.theta),
         ctr[1]+cam.radius*Math.cos(cam.phi)*Math.sin(cam.theta),
         ctr[2]+cam.radius*Math.sin(cam.phi)];
}

// ---------- picking ----------
function pick(x,y){
 const w=canvas.width,h=canvas.height;
 const fbo=gl.createFramebuffer(),tex=gl.createTexture(),rb=gl.createRenderbuffer();
 gl.bindTexture(gl.TEXTURE_2D,tex);
 gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
 gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
 gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
 gl.bindRenderbuffer(gl.RENDERBUFFER,rb);
 gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,w,h);
 gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,rb);
 gl.clearColor(1,1,1,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.bindBuffer(gl.ARRAY_BUFFER,S.idBuf);
 gl.vertexAttribPointer(locCol,3,gl.UNSIGNED_BYTE,true,0,0);
 gl.drawArrays(gl.POINTS,0,S.n);
 const px=new Uint8Array(4);
 gl.readPixels(x,h-y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);
 gl.bindFramebuffer(gl.FRAMEBUFFER,null);
 gl.deleteFramebuffer(fbo);gl.deleteTexture(tex);gl.deleteRenderbuffer(rb);
 gl.clearColor(0.07,0.07,0.08,1);
 const idx=px[0]|(px[1]<<8)|(px[2]<<16);
 if(idx>=S.n)return null;
 return idx;
}

// ---------- UI ----------
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");}
function badge(o){return '<span class="badge b-'+o+'">'+o+'</span>';}
const OUTCOME_NOTE={empty:"graph holds no such relation — NOT proof of absence",
 defer:"compiler/schema abstention", unknown:"insufficient completeness to assert",
 answer:"grounded answer with citations"};

function renderVariantBtns(){
 const el=document.getElementById("variantBtns");el.innerHTML="";
 const opts=Object.keys(S.p.variants).concat(["Human"]);
 for(const v of opts){
  const b=document.createElement("button");
  b.textContent=v==="Human"?"Human key":v;
  if(v===variant)b.classList.add("on");
  b.onclick=()=>{variant=v;refreshUI();recolor();};
  el.appendChild(b);
 }
 const vs=document.getElementById("variantStats");
 if(variant!=="Human"){
  const r=S.p.variants[variant];
  const sem=r.semantic_citation?(" · sem "+r.semantic_citation.accuracy):"";
  vs.textContent="uid-P "+r.micro_precision+" · uid-R "+r.micro_recall+sem+
   (r.entity_matches_at_05?(" · ent@0.5 "+r.entity_matches_at_05):"")+
   " · edges "+r.n_graph_edges;
 } else vs.textContent="human-verified key (reality, not the system)";
}
function renderQuestions(){
 const sel=document.getElementById("qSel");
 sel.innerHTML='<option value="">— none (free look) —</option>';
 for(const[id,kq]of Object.entries(S.p.key_questions)){
  const o=document.createElement("option");
  o.value=id;o.textContent=id+" — "+kq.question;
  sel.appendChild(o);
 }
 sel.value=qid;
}
function renderAnswer(){
 const el=document.getElementById("answerPanel");
 if(!qid){el.innerHTML="";return;}
 const kq=S.p.key_questions[qid];
 let h='<div class="panel">';
 h+='<b>'+esc(kq.question)+'</b>';
 h+='<div class="small">human expects: '+badge(kq.expected_outcome)+
    (kq.exhaustive?' <span class="small">(exhaustive)</span>':'')+'</div>';
 const labels=kq.candidate_labels||{};
 h+='<div><span class="small">Human answer:</span> '+
    (kq.expected_must_contain.length?kq.expected_must_contain.map(u=>
      '<span class="hit">'+esc(u)+'</span> <span class="small">'+
      esc(labels[u]||"")+'</span>').join(", "):"<i>empty</i>")+'</div>';
 if(variant!=="Human"){
  const qr=activeQuestionRecordFor(variant);
  if(qr){
   h+='<hr style="border-color:#333"><div><b>'+esc(variant)+'</b> '+
      badge(qr.actual_outcome)+' <span class="small">'+
      esc(OUTCOME_NOTE[qr.actual_outcome]||"")+'</span></div>';
   h+='<div>'+(qr.cited.length?qr.cited.map(c=>{
     const cls=c.status==="hit"?"hit":"wrong";
     const anon=c.unlabeled_segment?" (unlabeled segment)":"";
     return '<span class="'+cls+'">'+esc(c.uid)+'</span> <span class="small">'+
       esc(c.label)+anon+(c.matched_iou?(" iou "+c.matched_iou):"")+'</span>';
    }).join(", "):"<i>no citations</i>")+'</div>';
   if(qr.missed.length)h+='<div class="missed small">missed: '+
     qr.missed.map(esc).join(", ")+'</div>';
   h+='<div class="small">&ldquo;'+esc(qr.verbalized)+'&rdquo;</div>';
  } else h+='<div class="small">variant has no record for this question</div>';
 }
 h+='<div class="legend small" style="margin-top:6px">'+
  '<span class="hit">&#9632; hit</span><span class="wrong">&#9632; wrong/must-not</span>'+
  '<span class="missed">&#9632; missed</span><span class="anchor">&#9632; anchor class</span>'+
  '<span>&#9632; dimmed = uninvolved</span></div>';
 h+='</div>';
 el.innerHTML=h;
}
function renderPick(idx){
 const el=document.getElementById("pickPanel");
 if(idx===null){el.textContent="nothing selected";return;}
 const o=S.oracle[idx],p2=S.pred[idx];
 let h="";
 if(o>=0){
  const inf=S.p.objects[String(o)]||{};
  h+="<b>obj_"+o+"</b> — "+esc(inf.label||"?")+" <span class='small'>(Replica oracle id)</span><br>";
  if(inf.pred!==null&&inf.pred!==undefined){
   h+="C1 match: pred obj_"+inf.pred+" (iou "+inf.iou+")<br>";
   if(inf.c2)h+="C2 learned label: <b>"+esc(inf.c2)+"</b>"+
     (inf.c2!==inf.label?" <span class='wrong'>&ne; oracle</span>":" <span class='hit'>= oracle</span>")+"<br>";
  } else h+="C1: not matched (not recovered at IoU 0.5 or unproposed)<br>";
 } else h+="<i>no oracle object at this vertex</i><br>";
 if(p2>=0){
  const pi=S.p.preds[String(p2)]||{};
  h+="C1 instance: pred obj_"+p2+(pi.oracle!==null&&pi.oracle!==undefined?
    " (matched to obj_"+pi.oracle+")":" <i>(unlabeled segment)</i>");
 } else h+="C1 instance: unassigned vertex";
 el.innerHTML=h;
}
function refreshUI(){
 renderVariantBtns();renderQuestions();renderAnswer();
 document.getElementById("prov").innerHTML=
  "scene: <b>"+esc(S.p.scene_id)+"</b><br>git: "+esc(S.p.provenance.git_commit.slice(0,12))+
  "<br>key: "+esc(S.p.provenance.key.fixture_id)+
  (S.p.provenance.c1_bundle?("<br>C1 bundle sha: "+esc(S.p.provenance.c1_bundle.output_sha256.slice(0,16))+"…"):"")+
  "<br>c1 status: "+esc(S.p.c1_status)+
  "<br><span class='small'>"+esc(S.p.provenance.isolation)+"</span>";
}

let currentSid=null;
function setScene(sid){
 S=decodeScene(sid);
 if(sid!==currentSid){qid="";resetCam();}
 currentSid=sid;
 const sb=document.getElementById("sceneBtns");sb.innerHTML="";
 for(const s of DATA.scene_order){
  const b=document.createElement("button");b.textContent=s.replace("replica_","");
  if(s===sid)b.classList.add("on");
  b.onclick=()=>setScene(s);sb.appendChild(b);
 }
 const mb=document.getElementById("modeBtns");mb.innerHTML="";
 for(const[m,lab]of [["rgb","raw RGB"],["oracle","oracle instances"],["pred","C1 instances"]]){
  const b=document.createElement("button");b.textContent=lab;
  if(m===mode)b.classList.add("on");
  b.onclick=()=>{mode=m;setScene(sid);};mb.appendChild(b);
 }
 if(!(variant in S.p.variants)&&variant!=="Human")variant=Object.keys(S.p.variants)[0];
 refreshUI();recolor();
}

document.getElementById("qSel").onchange=e=>{qid=e.target.value;renderAnswer();recolor();};
document.getElementById("evalNote").textContent=DATA.evaluation_only;
document.getElementById("discs").innerHTML=DATA.disclosures.map(d=>
 '<div class="disc"><b>'+esc(d.title)+'.</b> '+esc(d.body)+'</div>').join("");

// ---------- input ----------
let drag=null;
canvas.addEventListener("mousedown",e=>{drag={x:e.clientX,y:e.clientY,moved:false,pan:e.shiftKey};});
window.addEventListener("mousemove",e=>{
 if(!drag)return;
 const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
 if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;
 if(drag.pan){
  const s=cam.radius*0.0015;
  const eye=eyePos(),ctr=cam.center;
  const fwd=norm3(sub3(ctr,eye)),right=norm3(cross3(fwd,[0,0,1])),up=cross3(right,fwd);
  cam.pan[0]+=(-dx*right[0]+dy*up[0])*s;
  cam.pan[1]+=(-dx*right[1]+dy*up[1])*s;
  cam.pan[2]+=(-dx*right[2]+dy*up[2])*s;
 }else{
  cam.theta-=dx*0.006;
  cam.phi=Math.min(1.5,Math.max(-1.5,cam.phi+dy*0.006));
 }
 drag.x=e.clientX;drag.y=e.clientY;draw();
});
window.addEventListener("mouseup",e=>{
 if(drag&&!drag.moved){
  const r=canvas.getBoundingClientRect();
  const idx=pick((e.clientX-r.left)*canvas.width/r.width,
                 (e.clientY-r.top)*canvas.height/r.height);
  renderPick(idx);draw();
 }
 drag=null;
});
canvas.addEventListener("wheel",e=>{e.preventDefault();
 cam.radius*=Math.exp(e.deltaY*0.001);
 cam.radius=Math.max(0.3,Math.min(cam.radius,80));draw();},{passive:false});
window.addEventListener("resize",draw);

setScene(DATA.scene_order[0]);
</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "runs" / "mvp_v1" / "viewer.html")
    ap.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    args = ap.parse_args(argv)

    scene_dirs = {s["scene_id"]: Path(s["room_dir"])
                  for s in json.loads(SCENE_MANIFEST.read_text())["scenes"]}
    manifest = json.loads(MANIFEST.read_text())["scenes"]

    payloads = []
    for sid in args.scenes:
        short = sid.replace("replica_", "")
        mvp_path = MVP_DIR / f"{sid}_mvp.json"
        if not mvp_path.exists():
            print(f"missing {mvp_path} — run `python3 tools/mvp_demo.py` first")
            return 1
        bundle = MS02_DIR / short
        meta = json.loads((bundle / "meta.json").read_text())
        if meta["output_sha256"] != manifest[short]["frozen_ms02_bundle"]["output_sha256"]:
            raise ValueError(f"{sid}: ms02 bundle hash mismatch vs manifest")
        payloads.append(build_scene_payload(
            scene_dirs[sid], bundle, sid, json.loads(mvp_path.read_text())))
        print(f"packed {sid}: {payloads[-1]['n']} vertices")

    html = build_viewer_html(payloads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({len(html)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
