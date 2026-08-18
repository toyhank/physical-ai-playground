"use client";

import { ContactShadows, OrbitControls, RoundedBox } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type SceneState = {
  robot: { joints: number[]; gripper_open: boolean };
  objects: { name: string; position: number[] }[];
  grasped: boolean;
  verified: boolean;
};

type Props = { sceneState: SceneState | null; running: boolean };
const HOME = [0.18, -0.92, 0.14, 1.52, 0, -0.74, 0];
const AXES: ("x" | "y" | "z")[] = ["y", "z", "y", "z", "y", "z", "y"];

function RobotLink({ length, slim = false }: { length: number; slim?: boolean }) {
  return <mesh position={[0, length / 2, 0]} castShadow receiveShadow>
    <capsuleGeometry args={[slim ? 0.064 : 0.085, length - (slim ? 0.1 : 0.14), 12, 24]} />
    <meshPhysicalMaterial color="#f5f6f2" roughness={0.27} metalness={0.08} clearcoat={0.35} />
  </mesh>;
}

function RobotJoint({ compact = false }: { compact?: boolean }) {
  return <group>
    <mesh rotation={[Math.PI / 2, 0, 0]} castShadow>
      <cylinderGeometry args={[compact ? 0.075 : 0.1, compact ? 0.075 : 0.1, compact ? 0.105 : 0.13, 32]} />
      <meshStandardMaterial color="#17201e" roughness={0.3} metalness={0.45} />
    </mesh>
    <mesh castShadow>
      <sphereGeometry args={[compact ? 0.082 : 0.108, 24, 16]} />
      <meshPhysicalMaterial color="#f2f3ef" roughness={0.25} clearcoat={0.4} />
    </mesh>
  </group>;
}

function PandaArm({ joints, gripperOpen, running }: { joints: number[]; gripperOpen: boolean; running: boolean }) {
  const jointRefs = useRef<(THREE.Group | null)[]>([]);
  const fingerLeft = useRef<THREE.Mesh>(null);
  const fingerRight = useRef<THREE.Mesh>(null);
  const targets = useMemo(() => HOME.map((home, index) => home + THREE.MathUtils.clamp(joints[index] ?? 0, -1.25, 1.25) * 0.42), [joints]);

  useFrame(({ clock }, delta) => {
    const blend = 1 - Math.exp(-delta * 6);
    jointRefs.current.forEach((joint, index) => {
      if (!joint) return;
      const axis = AXES[index];
      const idle = running && index > 3 ? Math.sin(clock.elapsedTime * 2.1 + index) * 0.018 : 0;
      joint.rotation[axis] = THREE.MathUtils.lerp(joint.rotation[axis], targets[index] + idle, blend);
    });
    const gap = gripperOpen ? 0.064 : 0.027;
    if (fingerLeft.current) fingerLeft.current.position.x = THREE.MathUtils.lerp(fingerLeft.current.position.x, -gap, blend);
    if (fingerRight.current) fingerRight.current.position.x = THREE.MathUtils.lerp(fingerRight.current.position.x, gap, blend);
  });
  const jointRef = (index: number) => (node: THREE.Group | null) => { jointRefs.current[index] = node; };

  return <group position={[-0.62, 0.16, 0.34]} scale={0.82}>
    <mesh castShadow receiveShadow><cylinderGeometry args={[0.2, 0.23, 0.22, 48]} /><meshPhysicalMaterial color="#e8ebe7" roughness={0.32} metalness={0.12} /></mesh>
    <mesh position={[0, 0.13, 0]} castShadow><cylinderGeometry args={[0.15, 0.17, 0.12, 40]} /><meshStandardMaterial color="#1b2421" roughness={0.28} metalness={0.5} /></mesh>
    <group ref={jointRef(0)} position={[0, 0.19, 0]}><RobotJoint /><RobotLink length={0.34} />
      <group ref={jointRef(1)} position={[0, 0.34, 0]}><RobotJoint /><RobotLink length={0.34} />
        <group ref={jointRef(2)} position={[0, 0.34, 0]}><RobotJoint /><RobotLink length={0.3} />
          <group ref={jointRef(3)} position={[0, 0.3, 0]}><RobotJoint compact /><RobotLink length={0.27} slim />
            <group ref={jointRef(4)} position={[0, 0.27, 0]}><RobotJoint compact /><RobotLink length={0.22} slim />
              <group ref={jointRef(5)} position={[0, 0.22, 0]}><RobotJoint compact /><RobotLink length={0.18} slim />
                <group ref={jointRef(6)} position={[0, 0.18, 0]}><RobotJoint compact />
                  <mesh position={[0, 0.09, 0]} castShadow><cylinderGeometry args={[0.09, 0.075, 0.18, 28]} /><meshStandardMaterial color="#202825" roughness={0.24} metalness={0.5} /></mesh>
                  <group position={[0, 0.2, 0]}>
                    <mesh ref={fingerLeft} position={[-0.064, 0.07, 0]} castShadow><boxGeometry args={[0.035, 0.18, 0.055]} /><meshStandardMaterial color="#111816" roughness={0.32} metalness={0.38} /></mesh>
                    <mesh ref={fingerRight} position={[0.064, 0.07, 0]} castShadow><boxGeometry args={[0.035, 0.18, 0.055]} /><meshStandardMaterial color="#111816" roughness={0.32} metalness={0.38} /></mesh>
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

function OpenContainer({ position }: { position: [number, number, number] }) {
  return <group position={position}>
    <RoundedBox args={[0.42, 0.045, 0.34]} radius={0.025} smoothness={4} receiveShadow castShadow><meshStandardMaterial color="#245fc1" roughness={0.34} /></RoundedBox>
    {[
      [0, 0.13, -0.16, 0.42, 0.24, 0.035], [0, 0.13, 0.16, 0.42, 0.24, 0.035],
      [-0.195, 0.13, 0, 0.035, 0.24, 0.3], [0.195, 0.13, 0, 0.035, 0.24, 0.3],
    ].map(([x, y, z, sx, sy, sz], index) => <RoundedBox key={index} position={[x, y, z]} args={[sx, sy, sz]} radius={0.014} smoothness={3} castShadow><meshPhysicalMaterial color="#2769d3" roughness={0.28} metalness={0.08} clearcoat={0.25} /></RoundedBox>)}
  </group>;
}

function Workcell({ sceneState, running }: Props) {
  const objects = Object.fromEntries((sceneState?.objects ?? []).map((item) => [item.name, item.position]));
  const world = (position: number[]): [number, number, number] => [position[0] * 1.8, 0.075, -position[1] * 1.8];
  const cubePosition = world(objects.red_cube ?? [-0.24, -0.04, 0.45]);
  const boxPosition = world(objects.blue_box ?? [0.22, 0.08, 0.45]);
  return <>
    <color attach="background" args={["#dce5df"]} /><fog attach="fog" args={["#dce5df", 3.4, 7]} />
    <ambientLight intensity={1.1} /><hemisphereLight args={["#f9fff7", "#65736c", 1.4]} />
    <spotLight position={[2.8, 4.2, 2.2]} angle={0.48} penumbra={0.7} intensity={65} castShadow shadow-mapSize={[1024, 1024]} /><directionalLight position={[-3, 2, -2]} intensity={1.8} />
    <RoundedBox position={[0, -0.11, 0]} args={[2.25, 0.2, 1.55]} radius={0.06} smoothness={4} receiveShadow castShadow><meshPhysicalMaterial color="#d8ccb5" roughness={0.68} metalness={0.03} /></RoundedBox>
    <gridHelper args={[2.1, 14, "#86978e", "#b8c1bb"]} position={[0, 0.005, 0]} />
    <PandaArm joints={sceneState?.robot.joints ?? []} gripperOpen={sceneState?.robot.gripper_open ?? true} running={running} />
    <RoundedBox position={cubePosition} args={[0.17, 0.17, 0.17]} radius={0.018} smoothness={4} castShadow receiveShadow><meshPhysicalMaterial color="#d94336" roughness={0.3} clearcoat={0.18} /></RoundedBox>
    <OpenContainer position={boxPosition} />
    <ContactShadows position={[0, 0.01, 0]} opacity={0.42} scale={3.2} blur={2.2} far={2.4} />
    <OrbitControls makeDefault target={[0, 0.62, 0]} minDistance={2.2} maxDistance={4.8} minPolarAngle={0.45} maxPolarAngle={1.48} />
  </>;
}

export default function RobotScene3D(props: Props) {
  return <div className="robotCanvas" aria-label="Interactive 3D view of a seven-axis robot arm">
    <Canvas shadows dpr={[1, 1.75]} camera={{ position: [2.55, 1.9, 2.45], fov: 38, near: 0.1, far: 20 }}><Workcell {...props} /></Canvas>
  </div>;
}
