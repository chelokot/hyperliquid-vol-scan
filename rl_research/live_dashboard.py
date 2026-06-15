"""Realtime live-trading dashboard. Reads the live_store (sqlite, WAL) and serves
a self-refreshing page: balance/equity per dex, positions, targets, model
predictions + premium, trade history, estimated fees, and retrain history."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live_store import LiveStore

store = LiveStore()

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>live</title>
<style>
body{background:#0b0e14;color:#cdd6f4;font:13px/1.5 ui-monospace,monospace;margin:0;padding:16px}
h1{font-size:15px;margin:0 0 8px;color:#89b4fa} .muted{color:#6c7086}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:14px}
.card{background:#11151c;border:1px solid #1e2430;border-radius:10px;padding:12px}
.big{font-size:26px;font-weight:700} .pos{color:#a6e3a1} .neg{color:#f38ba8} .flat{color:#6c7086}
table{width:100%;border-collapse:collapse} td,th{padding:3px 8px;text-align:right;border-bottom:1px solid #1e2430}
th{color:#7f849c;font-weight:600;text-align:right} td:first-child,th:first-child{text-align:left}
.tag{padding:1px 7px;border-radius:6px;font-size:11px}.dry{background:#45475a}.live{background:#a6334a}
</style></head><body>
<h1>🛰️ live engine <span id=mode></span> <span class=muted id=clock></span></h1>
<div class=grid id=top></div>
<div class=card><b>позиции / сигналы</b><table id=syms></table></div>
<div style=height:14px></div>
<div class=grid>
<div class=card><b>сделки (последние)</b><table id=trades></table></div>
<div class=card><b>ретрейны модели</b><table id=models></table></div>
</div>
<script>
let startEq=null;
const f=(x,d=2)=>x==null?'–':Number(x).toFixed(d);
const cls=v=>v>1e-9?'pos':v<-1e-9?'neg':'flat';
async function tick(){
 let s=await (await fetch('/api/state')).json();
 let tr=await (await fetch('/api/trades')).json();
 let md=await (await fetch('/api/models')).json();
 if(!s){document.getElementById('clock').textContent='нет данных';return;}
 document.getElementById('mode').innerHTML=s.live?'<span class="tag live">LIVE</span>':'<span class="tag dry">DRY-RUN</span>';
 const age=((Date.now()-s.ts_ms)/1000).toFixed(0);
 document.getElementById('clock').textContent=`sec ${s.second} | ${s.in_session?'СЕССИЯ':'вне сессии'} | обновлено ${age}s назад`;
 let eq=Object.values(s.dex_equity||{}).reduce((a,b)=>a+b,0);
 if(startEq===null&&eq>0)startEq=eq;
 let pnl=startEq?eq-startEq:0;
 let dexrows=Object.entries(s.dex_equity||{}).map(([d,v])=>`${d}: $${f(v)}`).join(' · ')||'–';
 let fees=tr.reduce((a,t)=>a+(t.notional||0),0);
 document.getElementById('top').innerHTML=`
  <div class=card><div class=muted>эквити (сумма dex)</div><div class=big>$${f(eq)}</div><div class=muted>${dexrows}</div></div>
  <div class=card><div class=muted>сессионный PnL</div><div class="big ${cls(pnl)}">${pnl>=0?'+':''}$${f(pnl)}</div><div class=muted>от старта дашборда $${f(startEq)}</div></div>
  <div class=card><div class=muted>оборот / оценка комиссий</div><div class=big>$${f(fees,0)}</div><div class=muted>${tr.length} сделок в логе</div></div>
  <div class=card><div class=muted>модель</div><div class=big>${md.length?'v'+md.length:'–'}</div><div class=muted>scale ${md[0]?md[0].summary.scale:'?'} · corr ${md[0]?md[0].summary.val_corr:'?'}</div></div>`;
 let sy=Object.entries(s.symbols||{}).map(([k,v])=>{
   let prem=(v.stock&&v.perp)?((v.stock/v.perp-1)*100):null;
   return `<tr><td>${k}</td><td class=${cls(v.target)}>${f(v.target,2)}</td><td class=${cls(v.szi)}>${f(v.szi,4)}</td>
   <td>${f(v.perp,3)}</td><td>${f(v.stock,3)}</td><td class=${cls(prem)}>${f(prem,3)}%</td>
   <td class=${cls(v.pred_bps)}>${f(v.pred_bps,2)}</td><td>$${f(v.notional,0)}</td><td>${v.ready?'✓':'…'}</td></tr>`;}).join('');
 document.getElementById('syms').innerHTML='<tr><th>пара<th>target<th>поз(szi)<th>perp<th>stock<th>премия<th>pred bps<th>нотионал<th>rdy</tr>'+sy;
 document.getElementById('trades').innerHTML='<tr><th>время<th>пара<th>сторона<th>размер<th>цена<th>нотионал<th>статус</tr>'+
   tr.slice(0,25).map(t=>`<tr><td>${new Date(t.ts_ms).toLocaleTimeString()}</td><td>${t.symbol}</td><td class=${t.side=='buy'?'pos':'neg'}>${t.side}</td><td>${f(t.size,4)}</td><td>${f(t.price,3)}</td><td>$${f(t.notional,0)}</td><td>${t.result||t.kind}</td></tr>`).join('');
 document.getElementById('models').innerHTML='<tr><th>время<th>val_corr<th>scale<th>rows<th>live</tr>'+
   md.slice(0,12).map(m=>`<tr><td>${new Date(m.ts_ms).toLocaleTimeString()}</td><td>${m.summary.val_corr}</td><td>${m.summary.scale}</td><td>${m.summary.rows||'–'}</td><td>${(m.summary.live_symbols||[]).length}</td></tr>`).join('');
}
setInterval(tick,1000);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
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
            elif self.path == "/api/trades":
                self._send(json.dumps(store.recent_trades(200)).encode(), "application/json")
            elif self.path == "/api/models":
                self._send(json.dumps(store.model_history(50)).encode(), "application/json")
            else:
                self.send_response(404); self.end_headers()
        except BrokenPipeError:
            pass

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    print(f"dashboard: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
