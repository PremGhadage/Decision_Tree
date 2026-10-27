import os
import pickle
import warnings

import numpy as np
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "decision.pkl"), "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
PLACED_IDX = list(model.classes_).index(1)
SCORES = ("ssc_p", "hsc_p", "degree_p", "mba_p")
LABELS = {
    "ssc_p": "10th grade score",
    "hsc_p": "12th grade score",
    "degree_p": "degree score",
    "mba_p": "MBA score",
}
# number of choices per categorical field (encoded 0..n-1, alphabetical order)
CHOICES = {"gender": 2, "ssc_b": 2, "hsc_b": 2, "hsc_s": 3,
           "degree_t": 3, "workex": 2, "specialisation": 2}
SL_NO = 108  # record number; not meaningful for a new student

app = Flask(__name__)


def pct(data, key, default=None):
    try:
        v = float(data.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"Enter a number for {LABELS.get(key, 'the employability test score')}.")
    if not 0 <= v <= 100:
        raise ValueError(f"{LABELS.get(key, 'Employability test score')} must be between 0 and 100.")
    return v


def read_input(data):
    row = {k: pct(data, k) for k in SCORES}
    row["etest_p"] = pct(data, "etest_p", 70)
    for k, n in CHOICES.items():
        try:
            v = int(data.get(k, 0))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid choice for {k}.")
        if not 0 <= v < n:
            raise ValueError(f"Invalid choice for {k}.")
        row[k] = v
    row["sl_no"] = SL_NO
    return row


def vec(row):
    return np.array([[row[f] for f in FEATURES]], dtype=float)


def placed(row):
    return int(model.predict(vec(row))[0]) == 1


def prob(row):
    return float(model.predict_proba(vec(row))[0][PLACED_IDX])


def band(p):
    if p >= 85: return "Very likely placed", "good"
    if p >= 65: return "Likely placed", "good"
    if p >= 45: return "Borderline", "warn"
    if p >= 25: return "Unlikely to be placed", "bad"
    return "Very unlikely to be placed", "bad"


def explain(row):
    X, t, out = vec(row), model.tree_, []
    for node in model.decision_path(X).indices:
        f = t.feature[node]
        if f < 0 or FEATURES[f] not in LABELS:
            continue
        val, thr = X[0, f], t.threshold[node]
        rel = "at or below" if val <= thr else "above"
        out.append(f"Your {LABELS[FEATURES[f]]} is {val:g}%, {rel} the {thr:.1f}% turning point the model checks here.")
    return out


def what_if(row, is_placed):
    step, tips = (-1 if is_placed else 1), []
    for k in SCORES:
        for d in range(1, 41):
            nv = row[k] + step * d
            if not 0 <= nv <= 100:
                break
            if placed(dict(row, **{k: nv})) != is_placed:
                tips.append((d, k, nv))
                break
    tips.sort()
    if is_placed:
        return "How safe is this result?", [
            f"It would flip if your {LABELS[k]} dropped about {d} points (to {nv:.0f}%)." for d, k, nv in tips[:3]]
    return "What could change it", [
        f"Raising your {LABELS[k]} by about {d} points (to {nv:.0f}%) could flip it to placed." for d, k, nv in tips[:3]]


@app.post("/api/predict")
def api_predict():
    try:
        row = read_input(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    p = prob(row) * 100
    is_placed = placed(row)
    label, tone = band(p)
    title, tips = what_if(row, is_placed)
    outcome = "would be placed" if is_placed else "would not be placed this time"
    imp = sorted(((LABELS[f], round(float(i) * 100, 1))
                  for f, i in zip(FEATURES, model.feature_importances_) if f in LABELS and i > 0),
                 key=lambda x: -x[1])
    return jsonify(
        placed=is_placed, probability=round(p, 1), label=label, tone=tone,
        summary=f"Based on these scores, the model predicts you {outcome}. "
                f"In the training data, students with a similar profile were placed about {p:.0f}% of the time.",
        reasons=explain(row), tips_title=title, tips=tips,
        importance=[{"name": n, "value": v} for n, v in imp])


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Placement Predictor</title>
<script>try{var t=localStorage.getItem("theme")||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");document.documentElement.dataset.theme=t}catch(e){document.documentElement.dataset.theme="light"}</script>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#f3f6f8;--card:#fff;--ink:#14202b;--mute:#5b6a77;--line:#dde4ea;--acc:#0b8a7a;--warn:#c98200;--bad:#d1434e;--track:#e5ebf0;color-scheme:light}
[data-theme=dark]{--bg:#0d1218;--card:#151c24;--ink:#e8eef3;--mute:#92a1ad;--line:#25303b;--acc:#2fc9b4;--warn:#f0b43c;--bad:#ff6b75;--track:#25303b;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:500 15px/1.5 Manrope,system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}
h1{font-size:1.5rem;margin:0;font-weight:800;letter-spacing:-.02em}header p{margin:2px 0 0;color:var(--mute)}
h2{font-size:1rem;margin:0 0 12px;font-weight:700}
button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;font-weight:700}
button:hover{border-color:var(--acc)}button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
[data-theme=dark] button.primary{color:#06201b}
:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.grid{display:grid;grid-template-columns:1fr 380px;gap:16px;align-items:start}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:16px}
.fld{margin-bottom:16px}.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
label{font-weight:700}small{color:var(--mute)}
input[type=range]{width:100%;accent-color:var(--acc);margin:6px 0 0}
input[type=number],select{font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 8px}
input[type=number]{width:84px;text-align:right}select{width:100%}
.num{display:flex;align-items:center;gap:4px}
.chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.chips .sp{flex:1}
details summary{cursor:pointer;font-weight:700}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:14px}
.opts label{display:block;margin-bottom:4px;font-size:.88rem}
.side{position:sticky;top:12px}@media(max-width:900px){.side{position:static}}
.ringbox{display:flex;align-items:center;gap:16px}
.ring{position:relative;width:120px;height:120px;flex:none}
.ring svg{transform:rotate(-90deg)}.ring circle{fill:none;stroke-width:10;stroke-linecap:round}
#arc{stroke:var(--acc);stroke-dasharray:326.7;stroke-dashoffset:326.7;transition:stroke-dashoffset .6s ease,stroke .3s}
#pct{position:absolute;inset:0;display:grid;place-items:center;font-size:1.6rem;font-weight:800}
#verdict{font-size:1.25rem;font-weight:800;margin:0}
ul{padding-left:18px;margin:6px 0 0}li{margin-bottom:6px}
.bar{display:grid;grid-template-columns:120px 1fr 44px;gap:8px;align-items:center;font-size:.85rem;margin-bottom:6px}
.bar i{height:8px;border-radius:8px;background:var(--acc);width:var(--w);display:block}
.bar div{background:var(--track);border-radius:8px}.bar b{text-align:right}
.err{color:var(--bad);font-weight:700}
.hist div{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);font-size:.88rem}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><div class="wrap">
<header><div><h1>Placement Predictor</h1><p>Enter your academic scores and see how likely you are to be placed.</p></div>
<button id="theme" aria-label="Switch light or dark theme">Theme</button></header>
<div class="grid"><main>
<section class="card"><h2>Your scores</h2>
<div class="chips" style="margin-bottom:16px"><span>Try a profile:</span>
<button data-p="strong">Strong</button><button data-p="average">Average</button><button data-p="weak">Needs work</button></div>
<div id="scores"></div>
<details><summary>More details (optional)</summary>
<p><small>Your current model doesn't use these fields, so they won't change the result.</small></p>
<div class="opts" id="opts"></div></details></section>
<div class="chips"><button class="primary" id="go">Predict</button><button id="reset">Reset</button><span class="sp"></span>
<label><input type="checkbox" id="live" checked> Update live as I adjust</label></div>
</main>
<aside class="side"><section class="card" aria-live="polite"><h2>Result</h2>
<div class="ringbox"><div class="ring"><svg width="120" height="120" viewBox="0 0 120 120"><circle cx="60" cy="60" r="52" stroke="var(--track)"/><circle id="arc" cx="60" cy="60" r="52"/></svg><div id="pct">--</div></div>
<div><p id="verdict">Waiting for input</p><small>Chance of being placed</small></div></div>
<p id="summary" style="margin-top:14px"></p>
<div id="body"></div></section>
<section class="card hist"><div class="row"><h2>Recent predictions</h2><button id="clr">Clear</button></div><div id="hist"><small>Nothing yet. Press Predict to save a result.</small></div></section>
</aside></div>
<p><small>This is a statistical estimate from past data, not a guarantee. Real hiring depends on many other factors.</small></p></div>
<script>
const $=s=>document.querySelector(s);
const SC=[["ssc_p","10th grade (SSC) score","Marks in your secondary school exam"],["hsc_p","12th grade (HSC) score","Marks in your higher secondary exam"],["degree_p","Degree score","Your undergraduate percentage"],["mba_p","MBA score","Your MBA percentage so far"]];
const OPT=[["gender","Gender",["Female","Male"]],["ssc_b","10th board",["Central","Others"]],["hsc_b","12th board",["Central","Others"]],["hsc_s","12th stream",["Arts","Commerce","Science"]],["degree_t","Degree type",["Commerce & Management","Others","Science & Technology"]],["workex","Work experience",["No","Yes"]],["specialisation","MBA specialisation",["Marketing & Finance","Marketing & HR"]]];
const PRE={strong:[82,78,75,72],average:[65,62,62,60],weak:[48,52,55,50]};
$("#scores").innerHTML=SC.map(([k,l,h])=>`<div class="fld"><div class="row"><label for="n_${k}">${l}</label><span class="num"><input id="n_${k}" type="number" min="0" max="100" step="0.1" value="65" inputmode="decimal">%</span></div><input id="r_${k}" type="range" min="0" max="100" step="0.1" value="65" aria-label="${l}"><small>${h}</small></div>`).join("");
$("#opts").innerHTML=OPT.map(([k,l,o])=>`<div><label for="o_${k}">${l}</label><select id="o_${k}">${o.map((x,i)=>`<option value="${i}">${x}</option>`).join("")}</select></div>`).join("")+`<div><label for="o_etest_p">Employability test score (%)</label><input id="o_etest_p" type="number" min="0" max="100" step="0.1" value="70" style="width:100%"></div>`;
let timer,seq=0;
function setScore(k,v){$("#n_"+k).value=v;$("#r_"+k).value=v}
SC.forEach(([k])=>{$("#r_"+k).oninput=e=>{$("#n_"+k).value=e.target.value;sched()};$("#n_"+k).oninput=e=>{$("#r_"+k).value=e.target.value;sched()}});
document.querySelectorAll("select,#o_etest_p").forEach(e=>e.onchange=sched);
function collect(){const d={};SC.forEach(([k])=>d[k]=$("#n_"+k).value);OPT.forEach(([k])=>d[k]=$("#o_"+k).value);d.etest_p=$("#o_etest_p").value;return d}
function sched(){if($("#live").checked){clearTimeout(timer);timer=setTimeout(()=>run(false),250)}}
async function run(save){
  const d=collect(),my=++seq;
  try{
    const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
    const r=await res.json();if(my!==seq)return;
    if(!res.ok){$("#summary").innerHTML=`<span class="err">${r.error||"Something went wrong."}</span>`;return}
    show(r);if(save)addHist(r,d);
  }catch(e){$("#summary").innerHTML='<span class="err">Could not reach the server. Check your connection and try again.</span>'}
}
function show(r){
  const c={good:"var(--acc)",warn:"var(--warn)",bad:"var(--bad)"}[r.tone];
  $("#arc").style.stroke=c;$("#arc").style.strokeDashoffset=326.7*(1-r.probability/100);
  $("#pct").textContent=Math.round(r.probability)+"%";
  $("#verdict").textContent=r.label;$("#verdict").style.color=c;$("#summary").textContent=r.summary;
  const mx=Math.max(...r.importance.map(i=>i.value),1);
  $("#body").innerHTML=`<h2>Why the model says this</h2><ul>${r.reasons.map(x=>`<li>${x}</li>`).join("")}</ul>
  <h2 style="margin-top:16px">${r.tips_title}</h2><ul>${(r.tips.length?r.tips:["No single score change within 40 points flips this result."]).map(x=>`<li>${x}</li>`).join("")}</ul>
  <h2 style="margin-top:16px">What the model weighs most</h2>${r.importance.map(i=>`<div class="bar"><span>${i.name}</span><div><i style="--w:${i.value/mx*100}%"></i></div><b>${i.value}%</b></div>`).join("")}`;
}
function getH(){try{return JSON.parse(localStorage.getItem("hist")||"[]")}catch(e){return[]}}
function drawH(h){$("#hist").innerHTML=h.length?h.map(x=>`<div><span>${x.l} (${x.p}%)</span><small>${x.s}</small></div>`).join(""):"<small>Nothing yet. Press Predict to save a result.</small>"}
function addHist(r,d){const h=getH();h.unshift({p:Math.round(r.probability),l:r.label,s:[d.ssc_p,d.hsc_p,d.degree_p,d.mba_p].join(" / ")});h.length=Math.min(h.length,5);try{localStorage.setItem("hist",JSON.stringify(h))}catch(e){}drawH(h)}
$("#clr").onclick=()=>{try{localStorage.removeItem("hist")}catch(e){}drawH([])};
$("#go").onclick=()=>run(true);
$("#reset").onclick=()=>{SC.forEach(([k])=>setScore(k,65));OPT.forEach(([k])=>$("#o_"+k).value=0);$("#o_etest_p").value=70;run(false)};
document.querySelectorAll("[data-p]").forEach(b=>b.onclick=()=>{PRE[b.dataset.p].forEach((v,i)=>setScore(SC[i][0],v));run(false)});
$("#theme").onclick=()=>{const t=document.documentElement.dataset.theme==="dark"?"light":"dark";document.documentElement.dataset.theme=t;try{localStorage.setItem("theme",t)}catch(e){}};
drawH(getH());run(false);
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=True)