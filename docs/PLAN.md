# Standalone voxel mod: plan

Goal: the Minecraft x Rocket League mechanics (building, digging, playing as Steve, sword / bow / TNT) without running
Minecraft, built so the same core can be dropped into other games.

## Layers

| Layer | Where | Game-specific? |
|---|---|---|
| Core: world grid, block list, raycast, place/break, saves, GPU data | `core/` | No |
| Renderer: the ReShade effect `MCPassthrough.fx` in voxel mode | friend kit (to be added under `rl/shaders/`) | No: needs only a camera and a depth buffer |
| Assets: atlas, block list, skin from VoxeLibre | `tools/assets/` -> `assets/` | No |
| Game adapter: camera in/out, input, collision, arena | `games/rocketleague/` + the BakkesMod plugin | Yes |

## Steps

1. **Grid and block list -> blocks show in RL through the voxel shader.**
   - Done here: `vox::World`, `vox::BlockRegistry`, the GPU export (`vox/gpu_layout.hpp`), the arena as voxels
     (`rl::buildSoccarArena`), the asset pipeline, and an offline preview (`tools/preview`) that renders the exported data
     with a port of the shader's voxel path.
   - Left: in the plugin, fill `MCVOX` / `MCHOLES` / `MCTYPES` from the core instead of from Minecraft's stream, and set
     `VoxActive`. Needs the plugin source (`MCBridge.cpp`, `compositor.cpp`) in this repo.
2. **Place and break from the car's camera.**
   - Done here: `vox::raycast`, `vox::breakBlock` / `vox::placeBlock`, `vox::Interactor` (Minecraft creative timing).
   - Left: feed it RL's camera and mouse buttons in the plugin; upload dirty rectangles each frame.
3. **Steve: movement, camera follow, hotbar.** Movement physics and collision against the grid; reuse the plugin's
   camera hook (behind its cvar); a hotbar overlay (the shader's overlay is off in voxel mode, so it needs a new one).
4. **Items:** sword, bow, TNT (the plugin already has melee / projectile / explosion messages to build on).
5. **Sounds and polish:** VoxeLibre's sounds (CC BY-SA) through the plugin.
6. **The friend's PC:** stream the world (`World::changesSince` journal) the way the kit already receives it.

## Open questions

- **RL units -> blocks.** `rl::ArenaMapping` defaults to 100 uu per block and RL's floor at y = 64. These must match the
  plugin's camera conversion; check against `MCBridge.cpp` / `arena.py` once they are in the repo.
- **The black screen** on the host while a friend is connected may come from the Minecraft drawing step (which
  standalone mode doesn't use) or from the network stream to the friend (which it still does). Test with the friend
  joining without connecting to the host's world first.
