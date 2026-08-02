"""Build a self-running captioned demo from the frozen MVP-v1 viewer.

The output is a presentation-only derivative.  It embeds the accepted viewer
unchanged, then appends a small guided-playback layer that drives the existing
scene, overlay, question, and answer-source controls.  It never recomputes an
answer or metric.

Usage:
  python3 tools/mvp_captioned_demo.py
  python3 tools/mvp_captioned_demo.py --autoplay-delay-ms 1200
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWER = REPO_ROOT / "runs" / "mvp_v1" / "viewer.html"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "mvp_v1" / "captioned_demo.html"
MARKER = "<!-- MVP-V1-CAPTIONED-DEMO -->"


_STYLE = r"""
<style id="guided-demo-style">
#guidedDemo{position:absolute;z-index:30;left:380px;right:20px;bottom:20px;
  color:#fff;pointer-events:none;font-family:-apple-system,'Segoe UI',sans-serif}
#guidedCaption{max-width:900px;margin:0 auto;background:rgba(12,14,18,.92);
  border:1px solid rgba(255,255,255,.22);border-radius:12px;padding:14px 16px;
  box-shadow:0 10px 35px rgba(0,0,0,.45);backdrop-filter:blur(7px)}
#guidedKicker{font-size:11px;text-transform:uppercase;letter-spacing:.12em;
  color:#79b8ff;font-weight:700;margin-bottom:4px}
#guidedTitle{font-size:18px;font-weight:700;line-height:1.25;margin-bottom:5px}
#guidedText{font-size:14px;line-height:1.45;color:#e8e8ea;max-width:78ch}
#guidedMeta{display:flex;align-items:center;gap:10px;margin-top:10px;color:#aaa;
  font-size:11px}
#guidedProgress{height:3px;background:#333;border-radius:2px;overflow:hidden;flex:1}
#guidedProgress>i{display:block;height:100%;width:0;background:#79b8ff}
#guidedControls{pointer-events:auto;display:flex;gap:6px;margin-top:9px;align-items:center;flex-wrap:wrap}
#guidedControls button{background:#242932;border:1px solid #566170;color:#fff;
  padding:5px 11px;border-radius:6px}
#guidedControls button:hover{background:#303846}
#guidedSource{margin-left:auto;color:#888;font-size:10px}
.guided-focus{outline:2px solid #79b8ff!important;outline-offset:3px}
@media(max-width:900px){
 #guidedDemo{left:374px;right:8px;bottom:8px}
 #guidedCaption{padding:10px 11px;border-radius:9px}
 #guidedTitle{font-size:15px}
 #guidedText{font-size:12px;line-height:1.35}
 #guidedControls{gap:4px}
 #guidedControls button{padding:4px 7px}
 #guidedSource{order:10;width:100%;margin-left:0;overflow-wrap:anywhere}
}
@media(max-width:760px){
 #guidedDemo{left:8px;right:8px;bottom:8px}
 #guidedCaption{max-width:none}
 #guidedText{font-size:12px;line-height:1.35}
 #guidedSource{display:none}
}
</style>
"""


_CONTROLS = r"""
<div id="guidedDemo" role="region" aria-label="Captioned MVP walkthrough">
  <div id="guidedCaption">
    <div id="guidedKicker">MVP-v1 guided demo</div>
    <div id="guidedTitle"></div>
    <div id="guidedText"></div>
    <div id="guidedMeta">
      <span id="guidedCount"></span>
      <span id="guidedProgress"><i></i></span>
      <span id="guidedTime"></span>
    </div>
    <div id="guidedControls">
      <button id="guidedPrev" type="button" aria-label="Previous demo step">Back</button>
      <button id="guidedPlay" type="button" aria-label="Pause guided demo">Pause</button>
      <button id="guidedNext" type="button" aria-label="Next demo step">Next</button>
      <button id="guidedRestart" type="button" aria-label="Restart guided demo">Restart</button>
      <span id="guidedSource"></span>
    </div>
  </div>
</div>
"""


_SCRIPT = r"""
<script id="guided-demo-script">
"use strict";
(() => {
 const STEPS = [
  {title:"A real captured 3D scene",duration:11000,scene:"replica_office_0",mode:"rgb",variant:"A",qid:"",orbit:true,focus:"sceneBtns",
   text:"This is the raw Replica office mesh: full-resolution colored vertices, not a diagram. The task is to answer structural questions and remain explicit about uncertainty."},
  {title:"Oracle objects: the evaluation reference",duration:9000,scene:"replica_office_0",mode:"oracle",variant:"A",qid:"",focus:"modeBtns",
   text:"Oracle colors show the dataset's semantic instances. They are used as an evaluation reference, not presented as learned perception."},
  {title:"C1: what the learned segmenter recovered",duration:11000,scene:"replica_office_0",mode:"pred",variant:"A",qid:"",focus:"modeBtns",
   text:"The frozen Mask3D instances replace oracle boundaries. Grey regions are unassigned; merged and missing objects expose the proposal-coverage ceiling inherited by every downstream answer."},
  {title:"Every visible claim has provenance",duration:10000,scene:"replica_office_0",mode:"rgb",variant:"A",qid:"",pickLabel:"sofa",focus:"pickPanel",
   text:"Click-to-inspect connects a raw point to its Replica id, oracle class, C1 match and IoU, and—where available—the C2 learned label."},
  {title:"The viewer does not invent missing metadata",duration:11000,scene:"replica_office_0",mode:"rgb",variant:"A",qid:"",pickOrphan:true,focus:"pickPanel",
   text:"This raw-only office sliver has no oracle object metadata. The viewer says exactly that while preserving the separate learned-instance attribution."},
  {title:"A: oracle boxes answer the table question",duration:10000,scene:"replica_office_0",mode:"oracle",variant:"A",qid:"Q05",focus:"variantBtns",
   text:"The question and human key stay fixed. Variant A supplies oracle boxes, and the graph returns grounded citations."},
  {title:"B: mesh-derived boxes retain the answer",duration:9000,scene:"replica_office_0",mode:"oracle",variant:"B",qid:"Q05",focus:"variantBtns",
   text:"Only the box source changes. The answer survives, which rules out box derivation as the cause of the later collapse."},
  {title:"C1: learned instances collapse the answer",duration:12000,scene:"replica_office_0",mode:"pred",variant:"C1",qid:"Q05",focus:"answerPanel",
   text:"With the same graph and reasoner, replacing instances drives office uid-recall to zero. This isolates the loss to learned instance extraction—not language or labels."},
  {title:"Room 2: C1 preserves the shelf anchor",duration:11000,scene:"replica_room_2",mode:"pred",variant:"C1",qid:"Q07",focus:"variantBtns",
   text:"On the shelf question, C1 uses the oracle label on a learned instance. The supporting shelf appears in blue and the answer retains grounded items."},
  {title:"C2: one label error removes the anchor",duration:13000,scene:"replica_room_2",mode:"pred",variant:"C2",qid:"Q07",focus:"answerPanel",
   text:"The instances are identical; only labels change. Shelf becomes vent, the blue support anchor disappears, and the answer becomes an honest empty rather than a fabricated citation."},
  {title:"The result is inspectable uncertainty",duration:15000,scene:"replica_room_2",mode:"oracle",variant:"Human",qid:"Q07",focus:"evalNote",scrollTo:"evalNote",
   text:"Human keys describe reality, outcome badges distinguish answer, empty, defer, and unknown, and disclosures bound every claim. The result is not solved raw-scene QA—it is a system that makes failure attributable and visible."}
 ];
 const el=id=>document.getElementById(id);
 const title=el("guidedTitle"), text=el("guidedText"), count=el("guidedCount");
 const time=el("guidedTime"), bar=el("guidedProgress").querySelector("i");
 const playBtn=el("guidedPlay"), side=el("side");
 let index=0, running=false, timer=null, progressFrame=null, orbitFrame=null, startedAt=0;

 function clearMotion(){
  if(timer!==null)clearTimeout(timer); timer=null;
  if(progressFrame!==null)cancelAnimationFrame(progressFrame); progressFrame=null;
  if(orbitFrame!==null)cancelAnimationFrame(orbitFrame); orbitFrame=null;
 }
 function clearFocus(){document.querySelectorAll(".guided-focus").forEach(x=>x.classList.remove("guided-focus"));}
 function focus(id){const x=el(id);if(x)x.classList.add("guided-focus");}
 function pickLabel(label){
  const entries=Object.entries(S.p.objects);
  const row=entries.find(([,v])=>v.label===label);
  if(!row)return renderPick(null);
  const oid=Number(row[0]);
  for(let i=0;i<S.n;i++)if(S.oracle[i]===oid){renderPick(i);return;}
  renderPick(null);
 }
 function pickOrphan(){for(let i=0;i<S.n;i++)if(S.oracle[i]<0){renderPick(i);return;}renderPick(null);}
 function startOrbit(){
  const tick=()=>{if(!running||!STEPS[index].orbit)return;cam.theta-=0.0012;draw();orbitFrame=requestAnimationFrame(tick);};
  orbitFrame=requestAnimationFrame(tick);
 }
 function setState(step){
  clearFocus();
  mode=step.mode||"rgb";
  setScene(step.scene);
  variant=step.variant||"A";
  qid=step.qid||"";
  resetCam();
  refreshUI();recolor();renderPick(null);
  side.scrollTop=0;
  if(step.pickLabel)pickLabel(step.pickLabel);
  if(step.pickOrphan)pickOrphan();
  if(step.scrollTo){const target=el(step.scrollTo);if(target)side.scrollTop=Math.max(0,target.offsetTop-18);}
  focus(step.focus);
  if(window.innerWidth<=760&&step.focus&&!step.scrollTo){
   const target=el(step.focus);
   if(target&&target.offsetTop>210)side.scrollTop=Math.max(0,target.offsetTop-205);
  }
  draw();
 }
 function paint(){
  const step=STEPS[index];
  title.textContent=step.title;text.textContent=step.text;
  count.textContent=(index+1)+" / "+STEPS.length;
  time.textContent=Math.round(step.duration/1000)+" sec";
  bar.style.width="0%";
  el("guidedSource").textContent="presentation-only · frozen viewer source: __SOURCE_SHA__";
  setState(step);
 }
 function animateProgress(){
  if(!running)return;
  const step=STEPS[index], pct=Math.min(100,(performance.now()-startedAt)/step.duration*100);
  bar.style.width=pct.toFixed(1)+"%";
  if(pct<100)progressFrame=requestAnimationFrame(animateProgress);
 }
 function schedule(){
  if(!running)return;
  const step=STEPS[index];startedAt=performance.now();
  progressFrame=requestAnimationFrame(animateProgress);
  if(step.orbit)startOrbit();
  timer=setTimeout(()=>{if(index===STEPS.length-1){pause();return;}index++;clearMotion();paint();schedule();},step.duration);
 }
 function play(){if(running)return;running=true;playBtn.textContent="Pause";playBtn.setAttribute("aria-label","Pause guided demo");schedule();}
 function pause(){if(!running)return;running=false;clearMotion();playBtn.textContent="Play";playBtn.setAttribute("aria-label","Play guided demo");}
 function go(next){pause();index=(next+STEPS.length)%STEPS.length;paint();}
 el("guidedPrev").onclick=()=>go(index-1);
 el("guidedNext").onclick=()=>go(index+1);
 el("guidedRestart").onclick=()=>{go(0);play();};
 playBtn.onclick=()=>running?pause():play();
 window.addEventListener("keydown",e=>{
  if(e.key===" "){e.preventDefault();running?pause():play();}
  else if(e.key==="ArrowRight")go(index+1);
  else if(e.key==="ArrowLeft")go(index-1);
 });
 window.__MVP_GUIDED_DEMO__={steps:STEPS,go,play,pause,get index(){return index;},get running(){return running;}};
 paint();
 if(!new URLSearchParams(location.search).has("autoplay")||new URLSearchParams(location.search).get("autoplay")!=="0"){
  setTimeout(play,__AUTOPLAY_DELAY_MS__);
 }else{playBtn.textContent="Play";playBtn.setAttribute("aria-label","Play guided demo");}
})();
</script>
"""


def build_captioned_demo(viewer_html: str, *, source_sha256: str,
                         autoplay_delay_ms: int = 1200) -> str:
    """Append deterministic guided playback to an accepted viewer."""
    if MARKER in viewer_html:
        raise ValueError("input already contains the captioned-demo layer")
    required = ("id=\"side\"", "id=\"variantBtns\"", "function setScene(",
                "function renderPick(", "setScene(DATA.scene_order[0]);")
    missing = [token for token in required if token not in viewer_html]
    if missing:
        raise ValueError(f"viewer is missing required contracts: {missing}")
    if autoplay_delay_ms < 0:
        raise ValueError("autoplay delay must be non-negative")
    script = (_SCRIPT.replace("__SOURCE_SHA__", source_sha256[:16])
              .replace("__AUTOPLAY_DELAY_MS__", str(autoplay_delay_ms)))
    return viewer_html + "\n" + MARKER + _STYLE + _CONTROLS + script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--viewer", type=Path, default=DEFAULT_VIEWER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--autoplay-delay-ms", type=int, default=1200)
    args = parser.parse_args(argv)

    viewer = args.viewer.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(viewer.encode("utf-8")).hexdigest()
    output = build_captioned_demo(
        viewer, source_sha256=source_sha,
        autoplay_delay_ms=args.autoplay_delay_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output, encoding="utf-8")
    print(f"source viewer sha256: {source_sha}")
    print(f"wrote {args.out} ({len(output)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
