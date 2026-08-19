"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { useLayoutEffect } from "react";
import * as THREE from "three";

type SceneState = {
  robot: { joints: number[]; finger_joints?: number[]; gripper_open: boolean };
  objects: { name: string; position: number[] }[];
  grasped: boolean;
  verified: boolean;
};

type Props = { sceneState: SceneState | null; running: boolean };

// MuJoCo is Z-up; Three.js is Y-up. This right-handed mapping preserves the
// simulator's X axis and maps (x, y, z) -> (x, z, -y).
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

function ArmMaterial({ dark = false }: { dark?: boolean }) {
  return <meshStandardMaterial color={dark ? "#1f2b28" : "#d1dbd6"} roughness={0.42} metalness={0.18} />;
}

function PlanarLink({ length, radius }: { length: number; radius: number }) {
  return <mesh position={[length / 2, 0, 0]} rotation={[0, 0, -Math.PI / 2]} castShadow receiveShadow>
    <capsuleGeometry args={[radius, length, 12, 28]} />
    <ArmMaterial />
  </mesh>;
}

function VerticalCylinder({ radius, halfHeight, dark = false }: { radius: number; halfHeight: number; dark?: boolean }) {
  return <mesh castShadow receiveShadow>
    <cylinderGeometry args={[radius, radius, halfHeight * 2, 32]} />
    <ArmMaterial dark={dark} />
  </mesh>;
}

function MuJoCoArm({ joints, fingerJoints, gripperOpen }: { joints: number[]; fingerJoints?: number[]; gripperOpen: boolean }) {
  const q = [...joints, 0, 0, 0, 0, 0, 0, 0].slice(0, 7);
  const fingers = fingerJoints ?? [gripperOpen ? 0.06 : 0, gripperOpen ? 0.06 : 0];

  return <group position={[0, 0.44, 0.37]}>
    <VerticalCylinder radius={0.11} halfHeight={0.07} dark />
    <group position={[0, 0.295, 0]} rotation={[0, q[0], 0]}>
      <PlanarLink length={0.33} radius={0.055} />
      <group position={[0.33, 0, 0]} rotation={[0, q[1], 0]}>
        <PlanarLink length={0.28} radius={0.05} />
        <group position={[0.28, 0, 0]} rotation={[0, q[2], 0]}>
          <PlanarLink length={0.18} radius={0.047} />
          <group position={[0.18, q[3], 0]}>
            <VerticalCylinder radius={0.05} halfHeight={0.055} dark />
            <group rotation={[q[4], 0, 0]}>
              <VerticalCylinder radius={0.047} halfHeight={0.045} />
              <group rotation={[0, 0, -q[5]]}>
                <VerticalCylinder radius={0.043} halfHeight={0.04} dark />
                <group rotation={[0, q[6], 0]}>
                  <VerticalCylinder radius={0.06} halfHeight={0.045} />
                  <VerticalCylinder radius={0.065} halfHeight={0.045} dark />
                  <group position={[0, 0, -fingers[0]]}>
                    <mesh position={[0, -0.035, -0.04]} castShadow receiveShadow>
                      <boxGeometry args={[0.09, 0.11, 0.02]} />
                      <ArmMaterial dark />
                    </mesh>
                  </group>
                  <group position={[0, 0, fingers[1]]}>
                    <mesh position={[0, -0.035, 0.04]} castShadow receiveShadow>
                      <boxGeometry args={[0.09, 0.11, 0.02]} />
                      <ArmMaterial dark />
                    </mesh>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
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

  return <>
    <MuJoCoCamera />
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
    <MuJoCoArm
      joints={sceneState?.robot.joints ?? [0.3, 0.8, -1, 0, 0, 0, 0]}
      fingerJoints={sceneState?.robot.finger_joints}
      gripperOpen={sceneState?.robot.gripper_open ?? true}
    />
    <RedCube position={cubePosition} />
    <BlueBox position={boxPosition} />
  </>;
}

export default function RobotScene3D(props: Props) {
  return <div className="robotCanvas" aria-label="MuJoCo-calibrated robot camera view">
    <div className="mujocoViewport">
      <Canvas shadows dpr={[1, 1.75]} camera={{ position: [0, 1.35, 1.15], fov: 49, near: 0.1, far: 20 }}>
        <Workcell {...props} />
      </Canvas>
    </div>
  </div>;
}
