"use client";

import { lazy, Suspense, useEffect, useRef, useState } from "react";

type TraceEvent = { kind: string; text: string; tone?: "good" | "bad" };
type SceneState = { frame:number; robot:{joints:number[];finger_joints?:number[];ee_position:number[];gripper_open:boolean}; objects:{name:string;position:number[]}[]; grasped:boolean; verified:boolean; physics:string; control_mode?:string; simulation_steps?:number; actuator_count?:number };
type StreamEvent = { type:string; text?:string; error?:string; name?:string; arguments?:Record<string,unknown>; result?:{success:boolean;error?:string}; state?:SceneState; verified?:boolean };
const RobotScene3D = lazy(() => import("./RobotScene3D"));
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const initialTrace: TraceEvent[] = [{kind:"SYSTEM",text:"Connecting to simulation backend…"}];
const browserDemoState: SceneState = {
  frame: 0,
  robot: {joints:[0.3,0.8,-1,0,0,0,0],finger_joints:[0.06,0.06],ee_position:[0,0,0.7],gripper_open:true},
  objects: [{name:"red_cube",position:[-0.24,-0.04,0.45]},{name:"blue_box",position:[0.22,0.08,0.45]}],
  grasped:false,
  verified:false,
  physics:"browser-demo",
  control_mode:"scripted-browser-demo",
  simulation_steps:0,
  actuator_count:0,
};
const browserPlan: TraceEvent[] = [
  {kind:"OBSERVE",text:"Captured 512 × 512 robot camera"},
  {kind:"MODEL",text:"Located red cube [557, 293] and blue box [547, 752]"},
  {kind:"TOOL",text:'set_gripper_state({"opened":true})'},
  {kind:"TOOL",text:'move({"x":293,"y":558,"high":true})'},
  {kind:"TOOL",text:'move({"x":293,"y":558,"high":false})'},
  {kind:"TOOL",text:'set_gripper_state({"opened":false})'},
  {kind:"TOOL",text:'move({"x":293,"y":558,"high":true})'},
  {kind:"TOOL",text:'move({"x":752,"y":547,"high":true})'},
  {kind:"TOOL",text:'move({"x":752,"y":547,"high":false})'},
  {kind:"TOOL",text:'set_gripper_state({"opened":true})'},
  {kind:"TOOL",text:'move({"x":752,"y":547,"high":true})'},
  {kind:"SUCCESS",text:"✓ Verified by browser demo state",tone:"good"},
];

function ClientRobotScene({sceneState,running}:{sceneState:SceneState|null;running:boolean}){
  const [mounted,setMounted]=useState(false);
  useEffect(()=>setMounted(true),[]);
  if(!mounted)return <div className="threeLoading">Preparing WebGL workcell…</div>;
  return <Suspense fallback={<div className="threeLoading">Loading 7-axis robot…</div>}><RobotScene3D sceneState={sceneState} running={running}/></Suspense>;
}

function describeEvent(event:StreamEvent):TraceEvent|null {
  if(event.type==="run_started") return {kind:"USER",text:event.text??"Task accepted"};
  if(event.type==="observe") return {kind:"OBSERVE",text:event.text??"Captured robot camera"};
  if(event.type==="model") return {kind:"MODEL",text:event.text??"Model response"};
  if(event.type==="tool") return {kind:"TOOL",text:`${event.name}(${JSON.stringify(event.arguments)})`};
  if(event.type==="result") return {kind:"RESULT",text:event.result?.success?"success":event.result?.error??"failed",tone:event.result?.success?"good":"bad"};
  if(event.type==="success") return {kind:"SUCCESS",text:"✓ Verified by simulator",tone:"good"};
  if(event.type==="failure") return {kind:"FAILURE",text:event.error??"Task failed",tone:"bad"};
  if(event.type==="stopped") return {kind:"STOPPED",text:"Stopped by user",tone:"bad"};
  return null;
}

export default function Home(){
  const [prompt,setPrompt]=useState("把红色方块放进蓝色盒子"); const [running,setRunning]=useState(false); const [online,setOnline]=useState(false); const [mode,setMode]=useState<"connecting"|"backend"|"browser">("connecting");
  const [sessionId,setSessionId]=useState<string|null>(null); const [sceneState,setSceneState]=useState<SceneState|null>(null); const [trace,setTrace]=useState<TraceEvent[]>(initialTrace); const [cameraVersion,setCameraVersion]=useState(0); const socketRef=useRef<WebSocket|null>(null); const demoToken=useRef(0);
  useEffect(()=>{let cancelled=false; async function connect(){try{const response=await fetch(`${API_URL}/api/sessions`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({seed:Math.floor(Math.random()*10000)})});if(!response.ok)throw new Error(`Backend returned ${response.status}`);const payload=await response.json();if(cancelled)return;setSessionId(payload.session_id);setSceneState(payload.state);setOnline(true);setMode("backend");setTrace([{kind:"SYSTEM",text:`Simulation ready · ${payload.state.physics}`},{kind:"OBSERVE",text:"Robot camera synchronized"}]);const socket=new WebSocket(`${API_URL.replace(/^http/,"ws")}/ws/${payload.session_id}`);socketRef.current=socket;socket.onmessage=(message)=>{const event=JSON.parse(message.data) as StreamEvent;if(event.type==="scene_state"&&event.state)setSceneState(event.state);const item=describeEvent(event);if(item)setTrace(items=>[...items,item]);if(event.type==="observe")setCameraVersion(value=>value+1);if(["success","failure","stopped"].includes(event.type))setRunning(false)};socket.onclose=()=>setOnline(false)}catch{if(cancelled)return;setSessionId("browser-demo");setSceneState(browserDemoState);setOnline(true);setMode("browser");setTrace([{kind:"SYSTEM",text:"Interactive browser demo ready · no API key required"},{kind:"OBSERVE",text:"Connect the Python backend for live MuJoCo physics"}])}}connect();return()=>{cancelled=true;demoToken.current+=1;socketRef.current?.close()}},[]);
  async function runDemo(){if(!prompt.trim()||!sessionId||running)return;setTrace(items=>[...items,{kind:"USER",text:prompt.trim()}]);setRunning(true);if(mode==="browser"){const token=++demoToken.current;for(let index=0;index<browserPlan.length;index+=1){await new Promise(resolve=>setTimeout(resolve,220));if(token!==demoToken.current)return;setTrace(items=>[...items,browserPlan[index]]);setSceneState(state=>state?{...state,frame:state.frame+6,robot:{...state.robot,gripper_open:index<5||index>=9}}:state)}setSceneState(state=>state?{...state,objects:state.objects.map(item=>item.name==="red_cube"?{...item,position:[0.22,0.08,0.45]}:item),verified:true}:state);setRunning(false);return}const response=await fetch(`${API_URL}/api/sessions/${sessionId}/run`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:prompt.trim()})});if(!response.ok){setRunning(false);setTrace(items=>[...items,{kind:"FAILURE",text:`Run rejected (${response.status})`,tone:"bad"}])}}
  async function stopDemo(){if(!sessionId)return;if(mode==="browser"){demoToken.current+=1;setTrace(items=>[...items,{kind:"STOPPED",text:"Stopped by user",tone:"bad"}]);setRunning(false);return}await fetch(`${API_URL}/api/sessions/${sessionId}/stop`,{method:"POST"});setRunning(false)}
  async function resetDemo(){if(!sessionId)return;demoToken.current+=1;if(mode==="browser"){setSceneState({...browserDemoState,objects:browserDemoState.objects.map(item=>({...item,position:[...item.position]}))});setRunning(false);setTrace([{kind:"SYSTEM",text:"Browser demo reset · ready for a new task"}]);return}await fetch(`${API_URL}/api/sessions/${sessionId}/reset`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({seed:Math.floor(Math.random()*10000)})});setRunning(false);setCameraVersion(value=>value+1);setTrace([{kind:"SYSTEM",text:"Scene reset · ready for a new task"}])}
  return <main className="shell">
    <header className="topbar"><div className="brand"><span className="brandMark">PA</span><div><h1>Physical AI Playground</h1><p>See how an embodied agent observes, decides, and acts.</p></div></div><div className="headerActions"><span className={`status ${online?"":"offline"}`}><i/> {mode==="backend"?"Simulation online":mode==="browser"?"Browser demo":"Connecting"}</span><button className="ghostButton" type="button" onClick={resetDemo} disabled={!online}>Reset scene</button></div></header>
    <section className="workspace"><div className="simulationPanel"><div className="panelHeading"><div><span className="eyebrow">LIVE SIMULATION</span><h2>MuJoCo-calibrated workspace</h2></div><div className="metrics"><span>{sceneState?.control_mode==="actuator-mj_step"?"MJ_STEP":"WEBGL 3D"}</span><span>{sceneState?.physics??"waiting"}</span><span>{sceneState?.actuator_count??0} ACT</span></div></div><div className={`scene ${running?"isRunning":""}`}><ClientRobotScene sceneState={sceneState} running={running}/><div className="sceneLegend"><span><i className="redDot"/> grasp target</span><span><i className="blueDot"/> place target</span><span>fixed MuJoCo camera · FOV 49°</span></div><div className="cameraTag">FRAME {sceneState?.frame??0} · STEP {sceneState?.simulation_steps??0}</div></div></div>
      <aside className="sideRail"><section className="cameraPanel"><div className="sectionTitle"><div><span className="eyebrow">MODEL VISION</span><h3>Robot camera</h3></div><span className="livePill">{online?"LIVE":"WAIT"}</span></div><div className="cameraFeed">{sessionId&&mode==="backend"?<img src={`${API_URL}/api/sessions/${sessionId}/camera.png?v=${cameraVersion}`} alt="The current RGB frame sent to the model"/>:<><div className="cameraCube"/><div className="cameraBox"/></>}<div className="crosshair">+</div><span>512 × 512 · user sees what the model sees</span></div></section><section className="tracePanel"><div className="sectionTitle"><div><span className="eyebrow">REASONING LOOP</span><h3>Agent trace</h3></div><span className="stepCount">{trace.length} events</span></div><div className="traceList">{trace.map((event,index)=><div className={`traceEvent ${event.tone??""}`} key={`${event.kind}-${index}`}><span className={`traceKind ${event.kind.toLowerCase()}`}>{event.kind}</span><p>{event.text}</p></div>)}{running&&<div className="traceEvent pending"><span className="pulse"/><p>Waiting for the next simulator observation</p></div>}</div></section></aside></section>
    <section className="commandBar"><div className="commandCopy"><span className="eyebrow">NATURAL LANGUAGE CONTROL</span><label htmlFor="task">Give the robot a task</label></div><div className="commandInput"><input id="task" value={prompt} onChange={event=>setPrompt(event.target.value)} onKeyDown={event=>{if(event.key==="Enter")runDemo()}}/><button className="runButton" type="button" onClick={runDemo} disabled={running||!online}>{running?"Running…":"Run task"}</button><button className="stopButton" type="button" onClick={stopDemo} disabled={!running}>Stop</button></div><p className="hint">支持中文和 English · Final success is verified from simulator ground truth.</p></section>
  </main>
}
