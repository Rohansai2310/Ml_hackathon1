"use strict";
const $ = (id) => document.getElementById(id);
let apiBase = "", result = null, busy = false, drag = null;
const svg = $("route-map"), NS = "http://www.w3.org/2000/svg";
let view = [0, 0, 1000, 650];
const percent = (n) => n == null ? "Unavailable" : `${(n * 100).toFixed(2)}%`;
const count = (n) => Number(n).toLocaleString();

function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function hideError() { $("error").hidden = true; }
async function request(path, data) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), data ? 190000 : 15000);
  try {
    const response = await fetch(apiBase + path, {
      method: data ? "POST" : "GET", cache: "no-store", signal: controller.signal,
      headers: data ? {"Content-Type": "application/json"} : {},
      body: data ? JSON.stringify(data) : undefined,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The server took too long to reply. Check the connection and server terminal, then retry.");
    throw error;
  } finally { clearTimeout(timeout); }
}

async function connect() {
  if (busy) return;
  hideError();
  $("match-button").disabled = true;
  $("reconnect").disabled = true;
  $("connection").textContent = "Connecting…";
  $("status-dot").className = "dot";
  try {
    const configResponse = await fetch("/client-config.json", {cache: "no-store"});
    if (!configResponse.ok) throw new Error("Client configuration could not be loaded.");
    const config = await configResponse.json();
    apiBase = config.api_base;
    $("server-address").textContent = config.server;
    const [health, catalog] = await Promise.all([request("/health"), request("/trips")]);
    if (!health.model_loaded) throw new Error("The server is reachable, but its model is not loaded. Start it with scripts/run_server.py.");
    const options = catalog.trips.map((id) => {
      const option = document.createElement("option"); option.value = id; option.textContent = `Trip ${id}`; return option;
    });
    $("trip-select").replaceChildren(...options);
    $("trip-select").disabled = !options.length;
    $("match-button").disabled = !options.length;
    $("trip-count").textContent = `${catalog.count} evaluation trips available on the server.`;
    if (!options.length) throw new Error("The server has no evaluator trips. Check its --data-root folder.");
    $("connection").textContent = "Server connected";
    $("status-dot").className = "dot online";
    $("progress").textContent = health.decoder_configured ? "Selected Random Forest + HMM ready." : "Server is using default HMM settings; load the selected decoder for the improved model.";
  } catch (error) {
    $("connection").textContent = "Connection needs attention";
    $("status-dot").className = "dot offline";
    $("trip-select").disabled = true;
    $("progress").textContent = "Check that both laptops share a network and the FastAPI server is running.";
    showError(error.message);
  } finally { $("reconnect").disabled = false; }
}

function clearResult() {
  result = null;
  $("download").disabled = $("download-map").disabled = $("folium-view").disabled = true;
  $("empty-state").hidden = false;
  svg.setAttribute("hidden", ""); $("folium-map").hidden = true;
  $("folium-map").removeAttribute("srcdoc");
  $("map-title").textContent = "Select a trip to begin";
  for (const id of ["hmm-accuracy", "rf-accuracy", "gps-count", "segment-count"]) $(id).textContent = "—";
  $("correct-points").textContent = "Valid reference GPS points";
  $("unresolved-count").textContent = "Unresolved references reported separately";
  $("elapsed").textContent = "Server processing time";
}

function element(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.appendChild(node); return node;
}
function setView() { svg.setAttribute("viewBox", view.join(" ")); }
function drawMap(geometry) {
  svg.replaceChildren();
  const points = geometry.gps;
  const cos = Math.cos(points[0][1] * Math.PI / 180);
  const origin = points[0];
  const world = ([lon, lat]) => [(lon - origin[0]) * cos, -(lat - origin[1])];
  const all = [...points, ...geometry.matched_segments.flat(2), ...geometry.reference_segments.flat(2)].map(world);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x,y] of all) { minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y); }
  const scale = Math.min(880 / Math.max(maxX - minX, 0.00001), 530 / Math.max(maxY - minY, 0.00001));
  const project = (p) => { const [x,y]=world(p); return [(x-(minX+maxX)/2)*scale+500,(y-(minY+maxY)/2)*scale+325]; };
  const path = (lines) => lines.map((line) => line.map((point, i) => `${i ? "L" : "M"}${project(point).map(n=>n.toFixed(2)).join(",")}`).join(" ")).join(" ");
  function layer(id, lines, color, width, opacity=1) {
    const group = element("g", {id: `layer-${id}`}, svg);
    element("path", {d: path(lines), fill:"none", stroke:color, "stroke-width":width, "stroke-opacity":opacity, "stroke-linecap":"round", "stroke-linejoin":"round", "vector-effect":"non-scaling-stroke"}, group);
    return group;
  }
  layer("roads",geometry.roads,"#c3cec8",1.4,.9);
  layer("reference",geometry.reference_segments.flat(),"#16834f",7,.55);
  layer("matched",geometry.matched_segments.flat(),"#e87532",3.5,.95);
  const gpsGroup = layer("gps",[points],"#3974d8",1.2,.65);
  for (let i=0;i<points.length;i+=Math.max(1,Math.floor(points.length/2000))) {
    const [cx,cy]=project(points[i]);
    const dot=element("circle",{cx,cy,r:1.8,fill:"#3974d8","fill-opacity":.8},gpsGroup);
    const title=element("title",{},dot);title.textContent=`GPS observation ${i}`;
  }
  for (const [point, color, label] of [[points[0],"#165b4e","Start"],[points.at(-1),"#b44837","End"]]) {
    const [cx,cy]=project(point); const dot=element("circle",{cx,cy,r:6,fill:color,stroke:"white","stroke-width":2},svg);
    element("title",{},dot).textContent=label;
  }
  view=[0,0,1000,650]; setView(); updateLayers();
}
function updateLayers() {
  document.querySelectorAll("[data-layer]").forEach(input=>{
    const group=$(`layer-${input.dataset.layer}`); if(group)group.style.display=input.checked?"":"none";
  });
}
function switchView(folium) {
  if (!result) return;
  svg.toggleAttribute("hidden", folium); $("folium-map").hidden = !folium;
  $("road-view").setAttribute("aria-pressed",String(!folium));
  $("folium-view").setAttribute("aria-pressed",String(folium));
  if (folium && !$("folium-map").hasAttribute("srcdoc")) $("folium-map").srcdoc=result.map_html;
  $("map-hint").textContent=folium?"Folium layers + street tiles need internet":"Scroll to zoom · drag to pan";
}

async function match() {
  if (busy) return;
  busy=true; hideError(); clearResult();
  $("match-button").disabled=$("trip-select").disabled=$("reconnect").disabled=true;
  const trip=$("trip-select").value;
  $("progress").textContent=`Matching trip ${trip} on the server, then evaluating its frozen predictions…`;
  $("map-title").textContent=`Processing trip ${trip}…`;
  try {
    result=await request("/evaluate",{trip_id:trip,partition:"evaluator_test"});
    const {evaluation:e,match:m}=result;
    $("map-title").textContent=`Trip ${m.trip_id}`;
    $("empty-state").hidden=true;
    drawMap(result.geometry);switchView(false);
    $("hmm-accuracy").textContent=percent(e.hmm_top1_accuracy);
    $("rf-accuracy").textContent=percent(e.rf_top1_accuracy);
    $("gps-count").textContent=count(m.number_of_gps_points);
    $("segment-count").textContent=count(m.route_segments.length);
    $("correct-points").textContent=e.status==="complete"?`${count(e.hmm_correct_points)} / ${count(e.known_points)} valid reference points`:e.reason;
    $("unresolved-count").textContent=e.status==="complete"?`${count(e.unresolved_points)} unresolved references excluded`:"Reference accuracy unavailable";
    $("elapsed").textContent=`${result.elapsed_seconds.toFixed(2)} s on the server`;
    $("result-note").textContent=e.note||e.reason;
    $("map-caption").textContent=`${count(m.number_of_candidates)} candidates · ${count(Object.keys(m.break_reasons).length)} recorded breaks`;
    $("progress").textContent="Complete. Explore the layers or download this result.";
    $("download").disabled=$("download-map").disabled=$("folium-view").disabled=false;
  } catch(error) {
    clearResult();showError(error.message);$("progress").textContent="The request did not complete. Check the server terminal, then retry.";
  } finally { busy=false;$("match-button").disabled=$("trip-select").disabled=$("reconnect").disabled=false; }
}
function download(content,type,name) {
  const url=URL.createObjectURL(new Blob([content],{type}));const link=document.createElement("a");
  link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("reconnect").addEventListener("click",connect);
$("match-button").addEventListener("click",match);
$("trip-select").addEventListener("change",clearResult);
$("layers").addEventListener("change",updateLayers);
$("road-view").addEventListener("click",()=>switchView(false));
$("folium-view").addEventListener("click",()=>switchView(true));
$("reset-map").addEventListener("click",()=>{view=[0,0,1000,650];setView();});
$("download").addEventListener("click",()=>{if(result){const {map_html,...json}=result;download(JSON.stringify(json,null,2),"application/json",`pathsnap-${result.match.trip_id}.json`);}});
$("download-map").addEventListener("click",()=>{if(result)download(result.map_html,"text/html",`pathsnap-${result.match.trip_id}.html`);});
svg.addEventListener("wheel",event=>{
  event.preventDefault();const factor=event.deltaY>0?1.15:.87;
  if(view[2]*factor<35||view[2]*factor>15000)return;
  const point=new DOMPoint(event.clientX,event.clientY).matrixTransform(svg.getScreenCTM().inverse());
  view=[point.x+(view[0]-point.x)*factor,point.y+(view[1]-point.y)*factor,view[2]*factor,view[3]*factor];setView();
},{passive:false});
svg.addEventListener("pointerdown",event=>{drag={x:event.clientX,y:event.clientY,view:[...view]};svg.setPointerCapture(event.pointerId);});
svg.addEventListener("pointermove",event=>{if(!drag)return;const scale=svg.getScreenCTM().a;view=[drag.view[0]-(event.clientX-drag.x)/scale,drag.view[1]-(event.clientY-drag.y)/scale,...drag.view.slice(2)];setView();});
svg.addEventListener("pointerup",()=>{drag=null;});
svg.addEventListener("pointercancel",()=>{drag=null;});
connect();
