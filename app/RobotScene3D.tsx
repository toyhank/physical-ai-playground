"use client";

import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

type RobotBody = { name: string; matrix: number[] };
type SceneState = { robot_id?: "panda"|"so101"; scene_profile?: "playground"|"vla_reference"; robot:{joints:number[];finger_joints?:number[];bodies?:RobotBody[];gripper_open:boolean}; objects:{name:string;position:number[]}[];verified:boolean };
type Props = { sceneState:SceneState|null; running:boolean };
type PandaPart={file:string;color:string};
type SO101Part={file:string;color:string;pos:[number,number,number];quat:[number,number,number,number]};
const WHITE="#fff",OFF_WHITE="#e6ebed",BLACK="#292d30",BLUE="#0a8ac7",GREEN="#00c853",YELLOW="#f5c327";
const API_URL=process.env.NEXT_PUBLIC_PUBLIC_DEMO==="true"?"/backend":process.env.NEXT_PUBLIC_API_URL??"http://127.0.0.1:8000";
const PANDA_ROOT="/models/franka_panda/assets",SO101_ROOT=`${API_URL}/models/so101/assets`,SO101_REFERENCE_ROOT=`${API_URL}/simulation-models/so101_vla_reference/assets`;

const PANDA_MESHES:Record<string,PandaPart[]>={
  link0:[["link0_0.obj",OFF_WHITE],["link0_1.obj",BLACK],["link0_2.obj",OFF_WHITE],["link0_3.obj",BLACK],["link0_4.obj",OFF_WHITE],["link0_5.obj",BLACK],["link0_7.obj",WHITE],["link0_8.obj",WHITE],["link0_9.obj",BLACK],["link0_10.obj",OFF_WHITE],["link0_11.obj",WHITE]].map(([file,color])=>({file,color})),
  link1:[{file:"link1.obj",color:WHITE}],link2:[{file:"link2.obj",color:WHITE}],
  link3:[["link3_0.obj",WHITE],["link3_1.obj",WHITE],["link3_2.obj",WHITE],["link3_3.obj",BLACK]].map(([file,color])=>({file,color})),
  link4:[["link4_0.obj",WHITE],["link4_1.obj",WHITE],["link4_2.obj",BLACK],["link4_3.obj",WHITE]].map(([file,color])=>({file,color})),
  link5:[["link5_0.obj",BLACK],["link5_1.obj",WHITE],["link5_2.obj",WHITE]].map(([file,color])=>({file,color})),
  link6:[["link6_0.obj",OFF_WHITE],["link6_1.obj",WHITE],["link6_2.obj",BLACK],["link6_3.obj",WHITE],["link6_4.obj",WHITE],["link6_5.obj",WHITE],["link6_6.obj",WHITE],["link6_7.obj",BLUE],["link6_8.obj",BLUE],["link6_9.obj",BLACK],["link6_10.obj",BLACK],["link6_11.obj",WHITE],["link6_12.obj",GREEN],["link6_13.obj",WHITE],["link6_14.obj",BLACK],["link6_15.obj",BLACK],["link6_16.obj",WHITE]].map(([file,color])=>({file,color})),
  link7:[["link7_0.obj",WHITE],["link7_1.obj",BLACK],["link7_2.obj",BLACK],["link7_3.obj",BLACK],["link7_4.obj",BLACK],["link7_5.obj",BLACK],["link7_6.obj",BLACK],["link7_7.obj",WHITE]].map(([file,color])=>({file,color})),
  hand:[["hand_0.obj",OFF_WHITE],["hand_1.obj",BLACK],["hand_2.obj",BLACK],["hand_3.obj",WHITE],["hand_4.obj",OFF_WHITE]].map(([file,color])=>({file,color})),
  left_finger:[["finger_0.obj",OFF_WHITE],["finger_1.obj",BLACK]].map(([file,color])=>({file,color})),right_finger:[["finger_0.obj",OFF_WHITE],["finger_1.obj",BLACK]].map(([file,color])=>({file,color})),
};
const so=(file:string,color:string,pos:[number,number,number],quat:[number,number,number,number]):SO101Part=>({file,color,pos,quat});
const SO101_MESHES:Record<string,SO101Part[]>={
  base:[so("base_motor_holder_so101_v1.stl",YELLOW,[-.00636471,-.00009944,-.0024],[.5,.5,.5,.5]),so("base_so101_v2.stl",YELLOW,[-.00636471,0,-.0024],[.5,.5,.5,.5]),so("sts3215_03a_v1.stl",BLACK,[.0263353,0,.0437],[1,0,0,0]),so("waveshare_mounting_plate_so101_v2.stl",YELLOW,[-.0309827,-.000199,.0474],[.5,.5,.5,.5])],
  shoulder:[so("sts3215_03a_v1.stl",BLACK,[-.0303992,.000422,-.0417],[.5,.5,.5,-.5]),so("motor_holder_so101_base_v1.stl",YELLOW,[-.0675992,-.000178,.01585],[.5,.5,-.5,.5]),so("rotation_pitch_so101_v1.stl",YELLOW,[.0122008,.000022,.0464],[.707107,-.707107,0,0])],
  upper_arm:[so("sts3215_03a_v1.stl",BLACK,[-.11257,-.0155,.0187],[0,-.707107,.707107,0]),so("upper_arm_so101_v1.stl",YELLOW,[-.065085,.012,.0182],[0,1,0,0])],
  lower_arm:[so("under_arm_so101_v1.stl",YELLOW,[-.06485,-.032,.0182],[0,1,0,0]),so("motor_holder_so101_wrist_v1.stl",YELLOW,[-.06485,-.032,.018],[0,-1,0,0]),so("sts3215_03a_v1.stl",BLACK,[-.1224,.0052,.0187],[0,0,1,0])],
  wrist:[so("sts3215_03a_no_horn_v1.stl",BLACK,[0,-.0424,.0306],[.5,.5,.5,-.5]),so("wrist_roll_pitch_so101_v2.stl",YELLOW,[0,-.028,.0181],[.5,-.5,-.5,-.5])],
  gripper:[so("sts3215_03a_v1.stl",BLACK,[.0077,.0001,-.0234],[.707107,-.707107,0,0]),so("wrist_roll_follower_so101_v1.stl",YELLOW,[0,-.000218,.00095],[0,1,0,0])],
  moving_jaw_so101_v1:[so("moving_jaw_so101_v1.stl",YELLOW,[0,0,.0189],[1,0,0,0])],
};
const reference=(file:string,color:string):SO101Part=>so(file,color,[0,0,0],[1,0,0,0]);
const SO101_REFERENCE_MESHES:Record<string,SO101Part[]>={
  base:[reference("Base.stl",OFF_WHITE),reference("Base_Motor.stl",BLACK)],
  shoulder:[reference("Rotation_Pitch.stl",OFF_WHITE),reference("Rotation_Pitch_Motor.stl",BLACK)],
  upper_arm:[reference("Upper_Arm.stl",OFF_WHITE),reference("Upper_Arm_Motor.stl",BLACK)],
  lower_arm:[reference("Lower_Arm.stl",OFF_WHITE),reference("Lower_Arm_Motor.stl",BLACK)],
  wrist:[reference("Wrist_Pitch_Roll.stl",OFF_WHITE),reference("Wrist_Pitch_Roll_Motor.stl",BLACK)],
  gripper:[reference("Fixed_Jaw.stl",OFF_WHITE),reference("Fixed_Jaw_Motor.stl",BLACK)],
  moving_jaw_so101_v1:[reference("Moving_Jaw.stl",OFF_WHITE)],
};
const HOME_BODIES:RobotBody[]=[
 {name:"link0",matrix:[1,0,0,0,0,1,0,.44,0,0,1,.37,0,0,0,1]},{name:"link1",matrix:[1,0,0,0,0,1,0,.773,0,0,1,.37,0,0,0,1]},
 {name:"link2",matrix:[.707,0,-.707,0,.707,0,.707,.773,0,-1,0,.37,0,0,0,1]},{name:"link3",matrix:[.707,-.707,0,-.223,.707,.707,0,.997,0,0,1,.37,0,0,0,1]},
 {name:"link4",matrix:[0,0,-1,-.165,-1,0,0,1.055,0,1,0,.37,0,0,0,1]},{name:"link5",matrix:[0,1,0,.219,-1,0,0,1.137,0,0,1,.37,0,0,0,1]},
 {name:"link6",matrix:[1,0,0,.219,0,0,-1,1.137,0,1,0,.37,0,0,0,1]},{name:"link7",matrix:[.707,0,.707,.307,0,-1,0,1.137,.707,0,-.707,.37,0,0,0,1]},
 {name:"hand",matrix:[1,0,0,.307,0,-1,0,1.03,0,0,-1,.37,0,0,0,1]},{name:"left_finger",matrix:[1,0,0,.307,0,-1,0,.972,0,0,-1,.41,0,0,0,1]},{name:"right_finger",matrix:[-1,0,0,.307,0,-1,0,.972,0,0,1,.33,0,0,0,1]},
];
function matrixFrom(v:number[]){return new THREE.Matrix4().set(...v as [number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number]);}
function toThree(p:number[]):[number,number,number]{return[p[0]??0,p[2]??0,-(p[1]??0)]}
function localMatrix(part:SO101Part){const[w,x,y,z]=part.quat;const r=new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(x,y,z,w));const c=new THREE.Matrix4().makeRotationX(-Math.PI/2),ci=new THREE.Matrix4().makeRotationX(Math.PI/2);return c.clone().multiply(r).multiply(ci).setPosition(...toThree(part.pos))}
function WorkcellCamera({robotId}:{robotId:string}){const{camera}=useThree();useLayoutEffect(()=>{const c=camera as THREE.PerspectiveCamera;if(robotId==="so101"){c.position.set(.58,.48,.58);c.up.set(0,1,0);c.lookAt(.18,.06,0)}else{c.position.set(0,1.35,1.15);c.up.set(0,.796,-.605);c.lookAt(0,.44,-.05)}c.fov=49;c.updateProjectionMatrix()},[camera,robotId]);return null}
function PandaMesh({part}:{part:PandaPart}){const src=useLoader(OBJLoader,`${PANDA_ROOT}/${part.file}`);const object=useMemo(()=>{const clone=src.clone(true);clone.traverse(child=>{if(child instanceof THREE.Mesh){child.material=new THREE.MeshStandardMaterial({color:part.color,roughness:.36,metalness:.12});child.castShadow=true;child.receiveShadow=true}});return clone},[src,part.color]);return <primitive object={object}/>}
function PandaBody({body}:{body:RobotBody}){const matrix=useMemo(()=>matrixFrom(body.matrix),[body.matrix]);return <group matrix={matrix} matrixAutoUpdate={false}><group rotation={[-Math.PI/2,0,0]}>{(PANDA_MESHES[body.name]??[]).map(p=><PandaMesh key={p.file} part={p}/>)}</group></group>}
function SO101Mesh({part,root}:{part:SO101Part;root:string}){const geometry=useLoader(STLLoader,`${root}/${part.file}`);const local=useMemo(()=>localMatrix(part),[part]);return <group matrix={local} matrixAutoUpdate={false}><mesh geometry={geometry} rotation={[-Math.PI/2,0,0]} castShadow receiveShadow><meshStandardMaterial color={part.color} roughness={.48} metalness={.05}/></mesh></group>}
function SO101Body({body,aligned}:{body:RobotBody;aligned:boolean}){const matrix=useMemo(()=>matrixFrom(body.matrix),[body.matrix]),meshes=aligned?SO101_REFERENCE_MESHES:SO101_MESHES,root=aligned?SO101_REFERENCE_ROOT:SO101_ROOT;return <group matrix={matrix} matrixAutoUpdate={false}>{(meshes[body.name]??[]).map((p,i)=><SO101Mesh key={`${p.file}-${i}`} part={p} root={root}/>)}</group>}
function RobotRenderer({robotId,bodies,aligned}:{robotId:string;bodies:RobotBody[];aligned:boolean}){return robotId==="so101"?<>{bodies.map(body=><SO101Body body={body} aligned={aligned} key={body.name}/>)}</>:<>{bodies.map(body=><PandaBody body={body} key={body.name}/>)}</>}
const COLORS:Record<string,string>={red_cube:"#e01a14",green_cube:"#16a34a",yellow_cube:"#e9a20b",purple_cube:"#8b35c8"};
function Cube({name,position,size}:{name:string;position:[number,number,number];size:number}){return <mesh position={position} castShadow receiveShadow><boxGeometry args={[size,size,size]}/><meshStandardMaterial color={COLORS[name]??"#777"} roughness={.42}/></mesh>}
function BlueBox({position,small,aligned}:{position:[number,number,number];small:boolean;aligned:boolean}){const width=aligned?.09:small?.14:.24,depth=aligned?.09:small?.12:.2,wall=aligned?.008:small?.012:.02,height=aligned?.024:small?.07:.11,wallY=aligned?.006:height/2;const walls=[[width/2-wall/2,wallY,0,wall,height,depth],[-width/2+wall/2,wallY,0,wall,height,depth],[0,wallY,depth/2-wall/2,width-2*wall,height,wall],[0,wallY,-depth/2+wall/2,width-2*wall,height,wall]];return <group position={position}><mesh castShadow receiveShadow><boxGeometry args={[width,aligned?.006:wall,depth]}/><meshStandardMaterial color="#0f43df"/></mesh>{walls.map((g,i)=><mesh key={i} position={[g[0],g[1],g[2]] as [number,number,number]} castShadow><boxGeometry args={[g[3],g[4],g[5]] as [number,number,number]}/><meshStandardMaterial color="#0f43df"/></mesh>)}</group>}
function Workcell({sceneState}:Props){const robotId=sceneState?.robot_id??"panda",small=robotId==="so101",aligned=sceneState?.scene_profile==="vla_reference";const objects=Object.fromEntries((sceneState?.objects??[]).map(o=>[o.name,o.position]));const bodies=sceneState?.robot.bodies?.length?sceneState.robot.bodies:(small?[]:HOME_BODIES);const target=(small?[.2,.06,0]:[0,.44,-.05]) as [number,number,number];return <><WorkcellCamera robotId={robotId}/><OrbitControls makeDefault target={target} enableDamping dampingFactor={.08} minDistance={.28} maxDistance={4} screenSpacePanning zoomToCursor/><color attach="background" args={["#ccd6cc"]}/><ambientLight intensity={.72}/><directionalLight position={[.2,2.4,.4]} intensity={2.2} castShadow/><mesh rotation={[-Math.PI/2,0,0]} position={[0,small?-.081:-.001,0]} receiveShadow><planeGeometry args={[6,6]}/><meshStandardMaterial color="#ccd6cc" roughness={.95}/></mesh><mesh position={[small ? .2 : 0,small?-.04:.4,0]} castShadow receiveShadow><boxGeometry args={[small ? .96 : 1.16,.08,small ? .72 : .92]}/><meshStandardMaterial color="#b89e7a" roughness={.7}/></mesh><RobotRenderer robotId={robotId} bodies={bodies} aligned={aligned}/>{Object.keys(objects).filter(n=>n.endsWith("_cube")).map(name=><Cube key={name} name={name} position={toThree(objects[name])} size={aligned?.024:small?.036:.06}/>)}<BlueBox position={toThree(objects.blue_box??[.3,.12,.006])} small={small} aligned={aligned}/></>}
export default function RobotScene3D(props:Props){const id=props.sceneState?.robot_id??"panda";return <div className="robotCanvas" aria-label={`Interactive ${id} MuJoCo view`}><div className="mujocoViewport"><Canvas shadows dpr={[1,1.75]} camera={{position:[.58,.48,.58],fov:49,near:.01,far:20}}><Workcell {...props}/></Canvas></div></div>}
