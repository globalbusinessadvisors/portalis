# Projectile Physics Demo - Video Storyboard

## Video Metadata
- **Title**: "Portalis: Python to WASM in NVIDIA Omniverse"
- **Duration**: 2:30 (2 minutes 30 seconds)
- **Target Audience**: Technical decision makers, simulation engineers, developers
- **Platform**: YouTube, NVIDIA Developer portal, Omniverse Exchange

---

## Scene Breakdown

### Scene 1: Opening (0:00 - 0:15)

**Visual**:
- Portalis logo on black background
- Smooth transition to title card
- Subtitle: "Real-time Physics Simulation with WASM"

**Audio**:
- Upbeat tech music (low volume)
- Voiceover: "Portalis brings Python simplicity to production-grade performance"

**Text Overlay**:
```
PORTALIS
Python → Rust → WASM → Everywhere
```

**Camera**: Static title cards with fade transitions

---

### Scene 2: Python Source Code (0:15 - 0:35)

**Visual**:
- Code editor showing `projectile.py`
- Highlight key functions:
  - `calculate_trajectory()`
  - `calculate_max_height()`
  - `calculate_range()`
- Syntax highlighting on

**Audio**:
- Voiceover: "Start with simple Python code. This physics simulation calculates projectile trajectories using basic mathematics."

**Text Overlay**:
```
Step 1: Write Python
Simple, readable, testable
```

**Camera**: Slow zoom into code, pan down to show functions

**Code Snippet to Show**:
```python
def calculate_trajectory(velocity, angle, time):
    """Calculate projectile position at time t"""
    angle_rad = math.radians(angle)
    vx = velocity * math.cos(angle_rad)
    vy = velocity * math.sin(angle_rad)

    x = vx * time
    y = vy * time - 0.5 * 9.81 * time**2

    return (x, y, 0.0)
```

---

### Scene 3: Portalis Translation (0:35 - 0:50)

**Visual**:
- Terminal window showing Portalis command
- Progress indicators
- Split screen: Python (left) → Rust (right)
- Quick transition showing generated Rust code

**Audio**:
- Terminal typing sounds
- Voiceover: "Portalis automatically translates Python to optimized Rust, then compiles to WebAssembly."

**Text Overlay**:
```
Step 2: Portalis Translation
Python → Rust → WASM
Automatic. Fast. Validated.
```

**Commands to Show**:
```bash
$ portalis translate projectile.py --target wasm
Analyzing Python source...
Generating Rust code...
Compiling to WASM...
✓ Complete: projectile.wasm (28 KB)
```

**Camera**: Terminal zoom, then split-screen comparison

---

### Scene 4: Rust Code Quality (0:50 - 1:05)

**Visual**:
- Show generated Rust code side-by-side with Python
- Highlight matching logic
- Show type safety and optimizations
- Display WASM module size

**Audio**:
- Voiceover: "The generated Rust code is production-ready, with strong typing, memory safety, and zero-cost abstractions."

**Text Overlay**:
```
Generated Rust Code
✓ Type-safe
✓ Memory-safe
✓ Optimized
✓ No runtime overhead
```

**Code Comparison**:
```rust
#[no_mangle]
pub extern "C" fn calculate_trajectory(
    velocity: f64,
    angle: f64,
    time: f64,
) -> (f64, f64, f64) {
    let angle_rad = angle * PI / 180.0;
    let vx = velocity * angle_rad.cos();
    let vy = velocity * angle_rad.sin();

    let x = vx * time;
    let y = vy * time - 0.5 * 9.81 * time * time;

    (x, y, 0.0)
}
```

**Camera**: Slow pan across code, pause on key sections

---

### Scene 5: NVIDIA Omniverse Setup (1:05 - 1:35)

**Visual**:
- Omniverse Create interface
- Load USD scene file
- Scene hierarchy showing:
  - Ground plane
  - Projectile sphere
  - WASM Physics Controller prim
- Select WASM controller, show attributes panel
- Highlight key attributes:
  - `wasmPath: projectile.wasm`
  - `entryFunction: update_physics`
  - `updateRate: 60.0`

**Audio**:
- UI click sounds
- Voiceover: "In NVIDIA Omniverse, the WASM module integrates seamlessly. Simply define a WASM controller primitive and link it to your physics objects."

**Text Overlay**:
```
Step 3: Omniverse Integration
Drop WASM into your scene
Configure. Connect. Run.
```

**Camera**: Screen recording with callouts highlighting key UI elements

---

### Scene 6: Real-time Simulation (1:35 - 2:05)

**Visual**:
- Press Play in Omniverse
- Projectile launches from launcher
- Smooth parabolic trajectory
- Camera follows projectile
- Performance overlay visible:
  - FPS counter: 60+ FPS
  - WASM execution: 3.2ms
  - Frame time: 16.6ms
- Trajectory trail shows perfect parabola
- Ground collision, bounce

**Audio**:
- Whoosh sound on launch
- Light background music continues
- Voiceover: "The simulation runs at over 60 frames per second, with the WASM module calculating physics in real-time. Latency is under 5 milliseconds."

**Text Overlay**:
```
Real-time Performance
60+ FPS | <5ms latency
WASM calculates every frame
```

**Camera**: Cinematic 3D camera following projectile, then wide shot showing full trajectory

**Performance HUD**:
```
FPS: 62
Frame Time: 16.1 ms
WASM Exec: 3.2 ms
Memory: 24 MB
```

---

### Scene 7: Performance Deep Dive (2:05 - 2:20)

**Visual**:
- Chart comparing Python vs WASM performance
- Bar graph:
  - Python (NumPy): 15 FPS, 65ms
  - Python (Pure): 8 FPS, 125ms
  - Portalis WASM: 60 FPS, 3ms
- Speedup factors: 4x faster, 20x faster
- Memory comparison chart
- Table of benchmark results

**Audio**:
- Voiceover: "Compared to native Python, the WASM module is up to 20 times faster, while using a fraction of the memory."

**Text Overlay**:
```
Benchmark Results
4-20x faster than Python
90% less memory
Production-ready performance
```

**Chart Data**:
```
Python (NumPy):     15 FPS    65 ms    120 MB
Python (Pure):       8 FPS   125 ms    180 MB
Portalis WASM:      62 FPS     3 ms     24 MB
                   -------   ------   -------
Speedup:            4-8x     20-40x      5x
```

---

### Scene 8: Use Cases & Conclusion (2:20 - 2:30)

**Visual**:
- Quick montage showing other use cases:
  - Robot arm IK solver
  - Warehouse digital twin
  - Sensor data processing
  - Fluid simulation
- Portalis logo returns
- Call-to-action screen

**Audio**:
- Voiceover: "From physics to robotics to digital twins, Portalis brings Python's ease to production performance. Try it today."

**Text Overlay**:
```
Industrial Applications
✓ Robotics & Kinematics
✓ Physics Simulation
✓ Sensor Processing
✓ Digital Twins

Visit: portalis.dev
GitHub: github.com/portalis
```

**Camera**: Fast cuts showing variety, end on logo

---

## Technical Requirements

### Video Specifications
- **Resolution**: 4K (3840×2160) or 1080p (1920×1080)
- **Frame Rate**: 60 FPS for smooth playback
- **Codec**: H.264 or H.265
- **Audio**: 44.1 kHz stereo

### Screen Recording Settings
- **Omniverse**: RTX Real-Time rendering enabled
- **Code Editor**: VS Code with Tomorrow Night theme
- **Terminal**: iTerm2 or Windows Terminal with Dracula theme
- **Font**: Fira Code or JetBrains Mono (ligatures on)

### Visual Effects
- **Transitions**: Smooth fades (0.5s)
- **Text Animations**: Slide-in from bottom
- **Code Highlighting**: Syntax-aware reveals
- **Performance Overlays**: Semi-transparent HUD

### Color Palette
- **Background**: Dark (#1a1a1a)
- **Accent**: NVIDIA Green (#76b900)
- **Text**: White (#ffffff) / Light Gray (#e0e0e0)
- **Code**: Syntax highlighting (Tomorrow Night)

### Music & Sound
- **Background Music**: Tech/corporate (royalty-free)
  - Suggestion: "Future Technology" by Scott Holmes
- **Sound Effects**: UI clicks, whoosh, subtle impacts
- **Voiceover**: Professional male or female voice, clear diction

---

## Script (Voiceover)

**[0:00 - 0:15] Opening**
> "Portalis brings Python simplicity to production-grade performance."

**[0:15 - 0:35] Python Source**
> "Start with simple Python code. This physics simulation calculates projectile trajectories using basic mathematics. Easy to write, easy to test, easy to understand."

**[0:35 - 0:50] Translation**
> "Portalis automatically translates Python to optimized Rust, then compiles to WebAssembly. The entire process takes seconds."

**[0:50 - 1:05] Rust Code**
> "The generated Rust code is production-ready, with strong typing, memory safety, and zero-cost abstractions. No manual porting required."

**[1:05 - 1:35] Omniverse Setup**
> "In NVIDIA Omniverse, the WASM module integrates seamlessly. Simply define a WASM controller primitive, link it to your physics objects, and you're ready to simulate."

**[1:35 - 2:05] Real-time Simulation**
> "The simulation runs at over 60 frames per second, with the WASM module calculating physics in real-time. Latency is under 5 milliseconds. What you're seeing isn't pre-baked animation—it's live computation."

**[2:05 - 2:20] Performance**
> "Compared to native Python, the WASM module is up to 20 times faster, while using a fraction of the memory. This is production-ready performance."

**[2:20 - 2:30] Conclusion**
> "From physics to robotics to digital twins, Portalis brings Python's ease to production performance. Try it today at portalis dot dev."

---

## Production Notes

### Recording Workflow

1. **Pre-production**
   - Set up Omniverse scene with proper lighting
   - Test WASM module performance
   - Prepare code examples in editor
   - Create performance charts

2. **Recording**
   - Record Omniverse simulation (OBS Studio)
   - Record code walkthroughs
   - Record terminal sessions
   - Capture performance data

3. **Post-production**
   - Edit in Premiere Pro or DaVinci Resolve
   - Add transitions and text overlays
   - Sync voiceover
   - Add background music
   - Color grade for consistency
   - Export in multiple resolutions

### Testing Checklist

- [ ] WASM module loads successfully
- [ ] Physics simulation runs smoothly (60+ FPS)
- [ ] Performance overlay displays correctly
- [ ] Code examples are syntax-highlighted
- [ ] Text overlays are readable at 1080p
- [ ] Audio levels are balanced
- [ ] No stuttering or dropped frames
- [ ] Video plays correctly on YouTube

### Distribution

- **YouTube**: Primary platform, SEO-optimized title and description
- **NVIDIA Developer**: Embedded on Omniverse Extension page
- **Portalis Website**: Featured on homepage
- **Social Media**: Short clips for Twitter, LinkedIn
- **Conferences**: Include in presentation decks

---

## Alternative Versions

### 30-second Trailer

Quick cut version for social media:
- 0:00-0:05: Python code
- 0:05-0:10: Portalis translation
- 0:10-0:20: Omniverse simulation
- 0:20-0:30: Performance + CTA

### Extended Tutorial (10 minutes)

Deep dive for developers:
- Detailed code walkthrough
- Step-by-step Omniverse setup
- Performance optimization tips
- Troubleshooting common issues
- Live Q&A segment

---

**Storyboard Version**: 1.0
**Created**: 2025-10-03
**Status**: Ready for production
