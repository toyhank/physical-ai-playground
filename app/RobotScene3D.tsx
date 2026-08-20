"use client";

import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

type RobotBody = { name: string; matrix: number[] };
type SceneState = {
  robot: { joints: number[]; finger_joints?: number[]; bodies?: RobotBody[]; gripper_open: boolean };
  objects: { name: string; position: number[] }[];
  grasped: boolean;
  verified: boolean;
};
type Props = { sceneState: SceneState | null; running: boolean };
type MeshPart = { file: string; color: string };

const WHITE = "#ffffff";
const OFF_WHITE = "#e6ebed";
const BLACK = "#404040";
const LIGHT_BLUE = "#0a8ac7";
const GREEN = "#00c853";
const ASSET_ROOT = "/models/franka_panda/assets";

const ROBOT_MESHES: Record<string, MeshPart[]> = {
  link0: [
    ["link0_0.obj", OFF_WHITE], ["link0_1.obj", BLACK], ["link0_2.obj", OFF_WHITE],
    ["link0_3.obj", BLACK], ["link0_4.obj", OFF_WHITE], ["link0_5.obj", BLACK],
    ["link0_7.obj", WHITE], ["link0_8.obj", WHITE], ["link0_9.obj", BLACK],
    ["link0_10.obj", OFF_WHITE], ["link0_11.obj", WHITE],
  ].map(([file, color]) => ({ file, color })),
  link1: [{ file: "link1.obj", color: WHITE }],
  link2: [{ file: "link2.obj", color: WHITE }],
  link3: [
    { file: "link3_0.obj", color: WHITE }, { file: "link3_1.obj", color: WHITE },
    { file: "link3_2.obj", color: WHITE }, { file: "link3_3.obj", color: BLACK },
  ],
  link4: [
    { file: "link4_0.obj", color: WHITE }, { file: "link4_1.obj", color: WHITE },
    { file: "link4_2.obj", color: BLACK }, { file: "link4_3.obj", color: WHITE },
  ],
  link5: [
    { file: "link5_0.obj", color: BLACK }, { file: "link5_1.obj", color: WHITE },
    { file: "link5_2.obj", color: WHITE },
  ],
  link6: [
    ["link6_0.obj", OFF_WHITE], ["link6_1.obj", WHITE], ["link6_2.obj", BLACK],
    ["link6_3.obj", WHITE], ["link6_4.obj", WHITE], ["link6_5.obj", WHITE],
    ["link6_6.obj", WHITE], ["link6_7.obj", LIGHT_BLUE], ["link6_8.obj", LIGHT_BLUE],
    ["link6_9.obj", BLACK], ["link6_10.obj", BLACK], ["link6_11.obj", WHITE],
    ["link6_12.obj", GREEN], ["link6_13.obj", WHITE], ["link6_14.obj", BLACK],
    ["link6_15.obj", BLACK], ["link6_16.obj", WHITE],
  ].map(([file, color]) => ({ file, color })),
  link7: [
    ["link7_0.obj", WHITE], ["link7_1.obj", BLACK], ["link7_2.obj", BLACK],
    ["link7_3.obj", BLACK], ["link7_4.obj", BLACK], ["link7_5.obj", BLACK],
    ["link7_6.obj", BLACK], ["link7_7.obj", WHITE],
  ].map(([file, color]) => ({ file, color })),
  hand: [
    { file: "hand_0.obj", color: OFF_WHITE }, { file: "hand_1.obj", color: BLACK },
    { file: "hand_2.obj", color: BLACK }, { file: "hand_3.obj", color: WHITE },
    { file: "hand_4.obj", color: OFF_WHITE },
  ],
  left_finger: [
    { file: "finger_0.obj", color: OFF_WHITE }, { file: "finger_1.obj", color: BLACK },
  ],
  right_finger: [
    { file: "finger_0.obj", color: OFF_WHITE }, { file: "finger_1.obj", color: BLACK },
  ],
};

const HOME_BODIES: RobotBody[] = [
  {name:"link0",matrix:[1,0,0,0,0,1,0,.44,0,0,1,.37,0,0,0,1]},
  {name:"link1",matrix:[1,0,0,0,0,1,0,.773,0,0,1,.37,0,0,0,1]},
  {name:"link2",matrix:[.7073883,0,-.7068252,0,.7068252,0,.7073883,.773,0,-1,0,.37,0,0,0,1]},
  {name:"link3",matrix:[.7073883,-.7068252,0,-.2233568,.7068252,.7073883,0,.9965347,0,0,1,.37,0,0,0,1]},
  {name:"link4",matrix:[-.0002037,0,-1,-.1649972,-1,0,.0002037,1.0548478,0,1,0,.37,0,0,0,1]},
  {name:"link5",matrix:[-.0002037,1,0,.2190196,-1,-.0002037,0,1.1372696,0,0,1,.37,0,0,0,1]},
  {name:"link6",matrix:[1,0,0,.2190196,0,0,-1,1.1372696,0,1,0,.37,0,0,0,1]},
  {name:"link7",matrix:[.7073883,0,.7068252,.3070196,0,-1,0,1.1372696,.7068252,0,-.7073883,.37,0,0,0,1]},
  {name:"hand",matrix:[.9999999,0,-.0003981,.3070196,0,-1,0,1.0302696,-.0003981,0,-.9999999,.37,0,0,0,1]},
  {name:"left_finger",matrix:[.9999999,0,-.0003981,.3070355,0,-1,0,.9718696,-.0003981,0,-.9999999,.41,0,0,0,1]},
  {name:"right_finger",matrix:[-.9999999,0,.0003981,.3070036,0,-1,0,.9718696,.0003981,0,.9999999,.33,0,0,0,1]},
];

function toThree(position: number[]): [number, number, number] {
  return [position[0] ?? 0, position[2] ?? 0, -(position[1] ?? 0)];
}

function MuJoCoCamera() {
  const { camera } = useThree();
  useLayoutEffect(() => {
    const perspective = camera as THREE.PerspectiveCamera;
    perspective.position.set(0, 1.35, 1.15);
    perspective.up.set(0, 0.796, -0.605);
    perspective.fov = 49;
    perspective.near = 0.1;
    perspective.far = 20;
    perspective.lookAt(0, 0.44, -0.05);
    perspective.updateProjectionMatrix();
  }, [camera]);
  return null;
}

function PandaMeshPart({ part }: { part: MeshPart }) {
  const source = useLoader(OBJLoader, `${ASSET_ROOT}/${part.file}`);
  const object = useMemo(() => {
    const clone = source.clone(true);
    clone.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshStandardMaterial({ color: part.color, roughness: 0.36, metalness: 0.12 });
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
    return clone;
  }, [source, part.color]);
  return <primitive object={object} />;
}

function PandaBody({ body }: { body: RobotBody }) {
  const matrix = useMemo(() => new THREE.Matrix4().set(...body.matrix as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]), [body.matrix]);
  return <group matrix={matrix} matrixAutoUpdate={false}>
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {(ROBOT_MESHES[body.name] ?? []).map((part) => <PandaMeshPart key={`${body.name}-${part.file}`} part={part} />)}
    </group>
  </group>;
}

function RedCube({ position }: { position: [number, number, number] }) {
  return <mesh position={position} castShadow receiveShadow>
    <boxGeometry args={[0.06, 0.06, 0.06]} />
    <meshStandardMaterial color="#e01a14" roughness={0.42} />
  </mesh>;
}

function BlueBox({ position }: { position: [number, number, number] }) {
  return <group position={position}>
    <mesh castShadow receiveShadow><boxGeometry args={[0.24, 0.024, 0.2]} /><meshStandardMaterial color="#0f43df" roughness={0.4} /></mesh>
    <mesh position={[0.11, 0.055, 0]} castShadow><boxGeometry args={[0.02, 0.11, 0.2]} /><meshStandardMaterial color="#0f43df" roughness={0.4} /></mesh>
    <mesh position={[-0.11, 0.055, 0]} castShadow><boxGeometry args={[0.02, 0.11, 0.2]} /><meshStandardMaterial color="#0f43df" roughness={0.4} /></mesh>
    <mesh position={[0, 0.055, -0.09]} castShadow><boxGeometry args={[0.2, 0.11, 0.02]} /><meshStandardMaterial color="#0f43df" roughness={0.4} /></mesh>
    <mesh position={[0, 0.055, 0.09]} castShadow><boxGeometry args={[0.2, 0.11, 0.02]} /><meshStandardMaterial color="#0f43df" roughness={0.4} /></mesh>
  </group>;
}

function Workcell({ sceneState }: Props) {
  const objects = Object.fromEntries((sceneState?.objects ?? []).map((item) => [item.name, item.position]));
  const cubePosition = toThree(objects.red_cube ?? [-0.2, 0.1, 0.47]);
  const boxPosition = toThree(objects.blue_box ?? [0.22, 0.12, 0.452]);
  const bodies = sceneState?.robot.bodies?.length ? sceneState.robot.bodies : HOME_BODIES;
  return <>
    <MuJoCoCamera />
    <OrbitControls
      makeDefault
      target={[0, 0.44, -0.05]}
      enableDamping
      dampingFactor={0.08}
      minDistance={0.45}
      maxDistance={4}
      minPolarAngle={0.08}
      maxPolarAngle={Math.PI * 0.94}
      screenSpacePanning
      zoomToCursor
    />
    <color attach="background" args={["#ccd6cc"]} />
    <ambientLight intensity={0.72} />
    <directionalLight position={[0, 2.4, 0.4]} intensity={2.2} castShadow shadow-mapSize={[1024, 1024]} />
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]} receiveShadow>
      <planeGeometry args={[6, 6]} />
      <meshStandardMaterial color="#ccd6cc" roughness={0.95} />
    </mesh>
    <mesh position={[0, 0.4, 0]} castShadow receiveShadow>
      <boxGeometry args={[1.16, 0.08, 0.92]} />
      <meshStandardMaterial color="#b89e7a" roughness={0.7} />
    </mesh>
    {bodies.map((body) => <PandaBody body={body} key={body.name} />)}
    <RedCube position={cubePosition} />
    <BlueBox position={boxPosition} />
  </>;
}

export default function RobotScene3D(props: Props) {
  return <div className="robotCanvas" aria-label="Interactive Franka Panda view: drag to rotate, scroll to zoom, and right-drag to pan">
    <div className="mujocoViewport">
      <Canvas shadows dpr={[1, 1.75]} camera={{ position: [0, 1.35, 1.15], fov: 49, near: 0.1, far: 20 }}>
        <Workcell {...props} />
      </Canvas>
    </div>
  </div>;
}
