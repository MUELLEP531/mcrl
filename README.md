# mcrl: Minecraft-style building in Rocket League

A voxel building mod for Rocket League that needs no Minecraft: blocks, digging and (later) playing as Steve, drawn
into RL's picture by a ReShade effect. It grew out of a setup that linked the real Minecraft to RL; this repo holds the
standalone version, built so the core can be dropped into other games. Offline, private matches and anti-cheat-off
play only.

## What's here

| Folder | What |
|---|---|
| `core/` | The game-agnostic C++17 voxel core: world grid, block list, raycast, place/break with Minecraft's creative timing, saves, and the data the shader reads. No game code. |
| `games/rocketleague/` | Rocket League's arena as voxels (the surfaces a player walks on and digs into). |
| `tools/assets/` | Builds the atlas, block list and skin from VoxeLibre's free textures into `assets/`. |
| `tools/preview/` | Renders exported worlds offline with a Python port of the shader's voxel path, to check everything without RL. |
| `assets/` | The built atlas (`mcrl_atlas.png`), block list (`mcrl_blocks.txt`) and skin (`mcrl_steve.png`). CC BY-SA: see `CREDITS.md`. |
| `docs/` | `SHADER_CONTRACT.md` (exactly what the effect reads) and `PLAN.md` (the steps and what's left). |

The BakkesMod plugin, the ReShade effect and the old Minecraft mod are not in this repo yet; connecting the core to
the plugin is the next step (see `docs/PLAN.md`).

## Build and test

Windows (Developer Command Prompt for Visual Studio) or Linux:

```
cmake -S . -B build
cmake --build build --config Debug
build\Debug\vox_tests.exe        (Linux: ./build/vox_tests)
```

Python tools (Python 3.10+, `pip install pillow numpy`):

```
python -m unittest discover -s tools/assets -p "test_*.py"
python -m unittest discover -s tools/preview -p "test_*.py"
```

## See it without Rocket League

```
build\Debug\vox_export.exe --blocks assets/mcrl_blocks.txt --out out/demo --atlas assets/mcrl_atlas.png --skin assets/mcrl_steve.png
python tools/preview/render.py out/demo --out out/demo.png
```

`vox_export` builds RL's arena with a hut, a tree, a few blocks and a hole dug into the floor; `render.py` draws it the
way the ReShade effect would (with a plain stand-in for RL's own picture).

## Rebuilding the assets

From VoxeLibre's texture folder (extracted flat):

```
python tools/assets/build_assets.py --textures <path to textures> --out assets
```
