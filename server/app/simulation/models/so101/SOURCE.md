# SO-101 model provenance

- Upstream: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- Imported commit: `7629d2ad9853d10fb903093a33ef6114099d97e5`
- Imported files: `so101_new_calib.xml`, mesh assets, upstream README, Apache-2.0 LICENSE
- Local modification: a wrist RGB camera was added to the gripper body; `scene.xml` is a playground-specific workcell.

The upstream model was generated from the official SO-101 CAD using
`onshape-to-robot`. Meshes are used for both the MuJoCo backend and the browser
renderer; no placeholder arm geometry is used.

