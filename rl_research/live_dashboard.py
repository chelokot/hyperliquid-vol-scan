"""Realtime live-trading dashboard + control plane. Reads live_store (sqlite WAL)
and serves a styled page (dark, canvas charts, no external deps): hero equity/PnL,
equity curve, control panel (enable pairs, per-pair leverage + budget weight,
pause/resume reduce-only, flatten), per-pair data cards, trade history, retrains.
Control writes go to the store; the engine applies them each second."""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live_store import LiveStore

store = LiveStore()

PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>HL live</title>
<style>
:root{--bg:#0a0d13;--card:#10141d;--card2:#0d1118;--bd:#1c2330;--tx:#c9d3e6;--mut:#5d677d;
--grn:#9ece6a;--red:#f7768e;--blu:#7aa2f7;--yel:#e0af68;--cy:#7dcfff}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#141a26 0,var(--bg) 55%);color:var(--tx);
font:13px/1.45 -apple-system,Inter,Segoe UI,sans-serif;-webkit-font-smoothing:antialiased}
.num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
header{display:flex;align-items:center;gap:10px;padding:12px 20px;border-bottom:1px solid var(--bd);position:sticky;top:0;
background:rgba(10,13,19,.88);backdrop-filter:blur(8px);z-index:5}
h1{font-size:15px;margin:0;font-weight:650} h1 .dot{color:var(--blu)}
.pill{font-size:11px;padding:3px 10px;border-radius:99px;font-weight:600;letter-spacing:.4px}
.live{background:linear-gradient(90deg,#f7768e,#bb4060);color:#fff}.dry{background:#2a3140;color:#9aa6bd}
.sess{background:#16351f;color:var(--grn)}.closed{background:#2a3140;color:var(--mut)}
.grow{flex:1}.age{color:var(--mut);font-size:12px}
button{font:inherit;cursor:pointer;border:1px solid var(--bd);background:#1a2130;color:var(--tx);border-radius:8px;padding:6px 12px;font-weight:600}
button:hover{border-color:#2c3650}
.btn-play{background:#16351f;color:var(--grn);border-color:#22512f}.btn-pause{background:#3a2417;color:var(--yel);border-color:#5a3a22}
.btn-flat{background:#3a1822;color:var(--red);border-color:#5a2433}
.btn-x{padding:3px 9px;font-size:11px;background:#2a1620;color:var(--red);border-color:#4a2030}
main{padding:18px 20px;max-width:1500px;margin:0 auto}
.hero{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
.metric{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--bd);border-radius:14px;padding:14px 16px}
.metric .l{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}
.metric .v{font-size:30px;font-weight:700;letter-spacing:-.5px}.metric .s{color:var(--mut);font-size:12px;margin-top:3px}
.pos{color:var(--grn)}.neg{color:var(--red)}.flat{color:var(--mut)}
.row{display:grid;grid-template-columns:1.6fr 1fr;gap:14px;margin-bottom:16px}
.panel{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px 16px;margin-bottom:16px}
.panel h2{font-size:12px;margin:0 0 12px;color:var(--mut);text-transform:uppercase;letter-spacing:.6px;font-weight:600;display:flex;align-items:center;gap:10px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.sym{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--bd);border-radius:12px;padding:12px}
.sym.off{opacity:.4}
.sym .hd{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.sym .nm{font-weight:700;font-size:14px}.sym .dex{color:var(--mut);font-size:10px;text-transform:uppercase}
.kv{display:flex;justify-content:space-between;font-size:12px;padding:2px 0}.kv .k{color:var(--mut)}
.gauge{height:6px;background:#0c0f16;border-radius:99px;margin:8px 0 4px;position:relative;overflow:hidden}
.gauge .mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#2c3444}
.gauge .fill{position:absolute;top:0;bottom:0;border-radius:99px}
table{width:100%;border-collapse:collapse}
td,th{padding:6px 8px;text-align:right;border-bottom:1px solid #161c27;white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
td:first-child,th:first-child{text-align:left}
.scroll{max-height:320px;overflow:auto}
canvas{width:100%;display:block}.empty{color:var(--mut);padding:26px;text-align:center}
input[type=number]{width:54px;background:#0c0f16;border:1px solid var(--bd);color:var(--tx);border-radius:6px;padding:3px 6px;font:inherit}
input[type=range]{width:90px;vertical-align:middle;accent-color:var(--blu)}
.sw{position:relative;width:38px;height:20px;display:inline-block}
.sw input{opacity:0;width:0;height:0}
.sl{position:absolute;inset:0;background:#2a3140;border-radius:99px;transition:.15s}
.sl:before{content:"";position:absolute;height:14px;width:14px;left:3px;top:3px;background:#8a93a6;border-radius:50%;transition:.15s}
.sw input:checked+.sl{background:#22512f}.sw input:checked+.sl:before{transform:translateX(18px);background:var(--grn)}
</style></head><body>
<header>
 <h1><span class=dot>◆</span> HL live</h1>
 <span id=mode class=pill></span><span id=sess class=pill></span>
 <button id=pause></button><button class=btn-flat id=flatall>⏹ закрыть всё</button>
 <span class=grow></span><span class=age id=clock></span>
</header>
<main>
 <div class=hero id=hero></div>
 <div class=row>
  <div class=panel style=margin-bottom:0><h2>эквити</h2><canvas id=eq height=170></canvas></div>
  <div class=panel style=margin-bottom:0><h2>модель / ретрейны</h2><div id=models></div></div>
 </div>
 <div class=panel><h2>управление · пары · бюджет · плечо</h2><div id=ctl></div></div>
 <div class=panel><h2>сигналы · предсказания</h2><div class=cards id=syms></div></div>
 <div class=panel><h2>сделки</h2><div class=scroll><table id=trades></table></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
const f=(x,d=2)=>(x==null||isNaN(x))?'–':Number(x).toFixed(d);
const sgn=v=>v>1e-9?'pos':v<-1e-9?'neg':'flat';
const money=x=>x==null?'–':'$'+Number(x).toLocaleString('en-US',{maximumFractionDigits:2});
let startEq=null, last=null;
const post=b=>fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(()=>setTimeout(refresh,150));

function line(cv,ys,color){const dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.height;cv.width=w*dpr;cv.style.height=h+'px';
 const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
 if(ys.length<2){g.fillStyle='#5d677d';g.font='12px sans-serif';g.fillText('ожидание данных…',14,h/2);return;}
 const p=8,lo=Math.min(...ys),hi=Math.max(...ys),r=(hi-lo)||1,X=i=>p+(w-2*p)*i/(ys.length-1),Y=v=>h-p-(h-2*p)*(v-lo)/r;
 const gr=g.createLinearGradient(0,0,0,h);gr.addColorStop(0,color+'40');gr.addColorStop(1,color+'00');
 g.beginPath();g.moveTo(X(0),Y(ys[0]));ys.forEach((v,i)=>g.lineTo(X(i),Y(v)));g.lineTo(X(ys.length-1),h-p);g.lineTo(X(0),h-p);g.fillStyle=gr;g.fill();
 g.beginPath();g.moveTo(X(0),Y(ys[0]));ys.forEach((v,i)=>g.lineTo(X(i),Y(v)));g.strokeStyle=color;g.lineWidth=1.8;g.stroke();
 g.fillStyle='#5d677d';g.font='10px ui-monospace';g.fillText(hi.toFixed(2),4,12);g.fillText(lo.toFixed(2),4,h-3);}
function spark(cv,ys,color){const dpr=devicePixelRatio||1,w=cv.clientWidth||200,h=cv.height;cv.width=w*dpr;cv.style.height=h+'px';
 const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);if(ys.length<2)return;
 const lo=Math.min(...ys),hi=Math.max(...ys),r=(hi-lo)||1;
 g.beginPath();ys.forEach((v,i)=>{const x=w*i/(ys.length-1),y=h-2-(h-4)*(v-lo)/r;i?g.lineTo(x,y):g.moveTo(x,y)});g.strokeStyle=color;g.lineWidth=1.4;g.stroke();}

function renderCtl(s){
 const rows=Object.entries(s.symbols||{}).map(([k,v])=>{
  const margin=v.leverage?v.notional/v.leverage:0;
  return `<tr data-k="${k}">
   <td>${k.split(':')[1]} <span class=dex>${k.split(':')[0]}</span></td>
   <td><label class=sw><input type=checkbox ${v.enabled?'checked':''} onchange="post({symbols:{'${k}':{enabled:this.checked}}})"><span class=sl></span></label></td>
   <td><input type=number min=1 max=20 value=${v.leverage} onchange="post({symbols:{'${k}':{leverage:+this.value}}})">x</td>
   <td><input type=range min=0 max=3 step=0.1 value=${v.weight} oninput="this.nextElementSibling.textContent=(+this.value).toFixed(1)" onchange="post({symbols:{'${k}':{weight:+this.value}}})"><span class=num>${(+v.weight).toFixed(1)}</span></td>
   <td class=num>${money(margin)} → ${money(v.notional)}</td>
   <td class="num ${sgn(v.szi)}">${f(v.szi,4)}</td>
   <td><button class=btn-x onclick="if(confirm('Закрыть ${k}?'))post({flatten:['${k}']})">✕</button></td></tr>`;}).join('');
 $('ctl').innerHTML='<table><tr><th>пара<th>вкл<th>плечо<th>бюджет(вес)<th>маржа→нотионал<th>позиция<th></tr>'+rows+'</table>'+
   '<div style="color:var(--mut);font-size:11px;margin-top:8px">бюджет каждого dex делится между включёнными парами пропорционально весу, затем умножается на плечо. pause = только сокращать (новое не открывается).</div>';
}

async function refresh(){
 let s,m,tr,md;
 try{[s,m,tr,md]=await Promise.all(['/api/state','/api/metrics','/api/trades','/api/models'].map(u=>fetch(u).then(r=>r.json())));}
 catch(e){$('clock').textContent='нет связи';return;}
 if(!s){$('clock').textContent='движок не запущен';$('hero').innerHTML='<div class=empty>движок ещё не писал состояние — запусти agents-hltrade-engine</div>';return;}
 $('mode').className='pill '+(s.live?'live':'dry');$('mode').textContent=s.live?'● LIVE':'DRY-RUN';
 $('sess').className='pill '+(s.in_session?'sess':'closed');$('sess').textContent=s.in_session?'СЕССИЯ':'вне сессии';
 const pb=$('pause');pb.textContent=s.paused?'▶ возобновить':'⏸ пауза';pb.className=s.paused?'btn-play':'btn-pause';
 $('clock').textContent='обновлено '+((Date.now()-s.ts_ms)/1000).toFixed(0)+'s · sec '+s.second;

 const eq=Object.values(s.dex_equity||{}).reduce((a,b)=>a+b,0);
 if(startEq===null&&eq>0)startEq=eq;
 const pnl=startEq?eq-startEq:0,pp=startEq?pnl/startEq*100:0;
 const open=Object.values(s.symbols||{}).filter(v=>Math.abs(v.szi)>1e-9).length;
 const expo=Object.values(s.symbols||{}).reduce((a,v)=>a+Math.abs((v.szi||0)*(v.perp||0)),0);
 const dexr=Object.entries(s.dex_equity||{}).map(([d,v])=>d+' '+money(v)).join(' · ')||'—';
 $('hero').innerHTML=`
  <div class=metric><div class=l>эквити (Σ dex)</div><div class="v num">${money(eq)}</div><div class=s>${dexr}${s.paused?' · ⏸ ПАУЗА':''}</div></div>
  <div class=metric><div class=l>сессионный PnL</div><div class="v num ${sgn(pnl)}">${pnl>=0?'+':''}${money(pnl)}</div><div class="s ${sgn(pnl)}">${pp>=0?'+':''}${f(pp)}% · старт ${money(startEq)}</div></div>
  <div class=metric><div class=l>позиции / экспозиция</div><div class="v num">${open}<span style=color:var(--mut);font-size:16px> / ${Object.keys(s.symbols||{}).length}</span></div><div class=s num>${money(expo)} нотионал</div></div>
  <div class=metric><div class=l>модель</div><div class="v num">${md.length?'v'+md.length:'baseline'}</div><div class=s>scale ${md[0]?md[0].summary.scale:'?'} · corr ${md[0]?md[0].summary.val_corr:'?'}</div></div>`;
 line($('eq'),m.map(x=>x.equity).filter(x=>x>0),'#7aa2f7');

 // control panel: render once, then only when the symbol set / enabled / lev / weight changed (don't clobber edits)
 const sig=JSON.stringify(Object.entries(s.symbols||{}).map(([k,v])=>[k,v.enabled,v.leverage,v.weight]));
 if(sig!==last){renderCtl(s);last=sig;}
 else Object.entries(s.symbols||{}).forEach(([k,v])=>{const r=document.querySelector(`tr[data-k="${k}"] td:nth-child(6)`);if(r){r.textContent=f(v.szi,4);r.className='num '+sgn(v.szi);}});

 const syms=Object.entries(s.symbols||{});
 $('syms').innerHTML=syms.map(([k,v])=>{const prem=(v.stock&&v.perp)?((v.stock/v.perp-1)*100):null;
  const tw=Math.min(Math.abs(v.target),1)*50,fill=v.target>=0?`left:50%;width:${tw}%;background:var(--grn)`:`right:50%;width:${tw}%;background:var(--red)`;
  return `<div class="sym ${v.enabled?'':'off'}"><div class=hd><span class=nm>${k.split(':')[1]}</span><span class=dex>${k.split(':')[0]} · ${v.ready?'ready':'…'}</span></div>
   <div class=gauge><div class=mid></div><div class=fill style="${fill}"></div></div>
   <div class=kv><span class=k>target</span><span class="num ${sgn(v.target)}">${f(v.target,2)}</span></div>
   <div class=kv><span class=k>позиция</span><span class="num ${sgn(v.szi)}">${f(v.szi,4)}</span></div>
   <div class=kv><span class=k>perp / stock</span><span class=num>${f(v.perp,2)} / ${f(v.stock,2)}</span></div>
   <div class=kv><span class=k>премия</span><span class="num ${sgn(prem)}">${f(prem,3)}%</span></div>
   <div class=kv><span class=k>прогноз 60с</span><span class="num ${sgn(v.pred_bps)}">${f(v.pred_bps,2)} bps</span></div>
   <canvas height=30 data-sym="${k}"></canvas></div>`;}).join('');
 syms.forEach(([k])=>{const cv=document.querySelector(`canvas[data-sym="${k}"]`);if(cv)spark(cv,m.map(x=>x.syms&&x.syms[k]?x.syms[k].prem:null).filter(x=>x!=null),'#7dcfff');});

 $('trades').innerHTML='<tr><th>время<th>пара<th>сторона<th>размер<th>цена<th>нотионал<th>статус</tr>'+
  (tr.length?tr.slice(0,60).map(t=>`<tr><td class=num>${new Date(t.ts_ms).toLocaleTimeString()}</td><td>${t.symbol.split(':')[1]}</td><td class=${t.side=='buy'?'pos':(t.side=='sell'?'neg':'flat')}>${t.side}</td><td class=num>${f(t.size,4)}</td><td class=num>${f(t.price,3)}</td><td class=num>${money(t.notional)}</td><td class=flat>${t.result||t.kind}</td></tr>`).join(''):'<tr><td colspan=7 class=empty>сделок ещё нет</td></tr>');
 $('models').innerHTML=md.length?'<table><tr><th>время<th>corr<th>scale<th>rows<th>live</tr>'+md.slice(0,10).map(x=>`<tr><td class=num>${new Date(x.ts_ms).toLocaleTimeString()}</td><td class="num pos">${x.summary.val_corr}</td><td class=num>${x.summary.scale}</td><td class=num>${(x.summary.rows||0).toLocaleString()}</td><td class=num>${(x.summary.live_symbols||[]).length}</td></tr>`).join('')+'</table>':'<div class=empty>ретрейнов ещё нет<br><span style=font-size:11px>baseline из production_ensemble.pt</span></div>';
}
$('pause').onclick=()=>fetch('/api/state').then(r=>r.json()).then(s=>post({paused:!(s&&s.paused)}));
$('flatall').onclick=()=>{if(confirm('Закрыть ВСЕ позиции?'))post({flatten:['ALL']})};
setInterval(refresh,1000);refresh();addEventListener('resize',refresh);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            if self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(json.dumps(store.latest_state()).encode(), "application/json")
            elif self.path == "/api/metrics":
                self._send(json.dumps(store.load_metrics(720)).encode(), "application/json")
            elif self.path == "/api/trades":
                self._send(json.dumps(store.recent_trades(200)).encode(), "application/json")
            elif self.path == "/api/models":
                self._send(json.dumps(store.model_history(50)).encode(), "application/json")
            else:
                self.send_response(404); self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        if self.path != "/api/control":
            self.send_response(404); self.end_headers(); return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            ctrl = store.get_control() or {"paused": False, "flatten": [], "symbols": {}}
            if "paused" in body:
                ctrl["paused"] = bool(body["paused"])
            if "symbols" in body:
                for sym, patch in body["symbols"].items():
                    ctrl.setdefault("symbols", {}).setdefault(sym, {}).update(patch)
            if "flatten" in body:
                ctrl["flatten"] = sorted(set((ctrl.get("flatten") or []) + list(body["flatten"])))
            store.set_control(ctrl)
            self._send(json.dumps({"ok": True}).encode(), "application/json")
        except Exception as exc:
            self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

    def log_message(self, *args) -> None:
        pass


class DualStackServer(ThreadingHTTPServer):
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("DASH_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DASH_PORT", "8787")))
    ap.add_argument("--cert", default=os.environ.get("DASH_TLS_CERT"))
    ap.add_argument("--key", default=os.environ.get("DASH_TLS_KEY"))
    args = ap.parse_args()
    DualStackServer.address_family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    server = DualStackServer((args.host, args.port), Handler)
    scheme = "http"
    if args.cert and args.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"dashboard: {scheme}://[{args.host}]:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
