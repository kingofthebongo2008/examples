uniform vec4 Front;
uniform vec4 Back;

void main()
{
     gl_FragColor = gl_FrontFacing ? Front : Back;
}