#version 410 core
//
// pass: shadow / depth-only
//
// No color writes — OpenGL populates the depth attachment automatically
// from ``gl_FragCoord.z``. We still need a frag shader present, but it
// has no work to do.

void main() {
}
