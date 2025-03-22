import OpenGL.GL as gl
import OpenGL.GLUT as glut

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glVertex2f(-0.5, -0.5)
    gl.glVertex2f(0.0, 0.5)
    gl.glVertex2f(0.5, -0.5)
    gl.glEnd()
    gl.glFlush()

def main():
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB)
    glut.glutInitWindowSize(400, 400)
    glut.glutCreateWindow("Hello, OpenGL!")
    glut.glutDisplayFunc(display)
    glut.glutMainLoop()

if __name__ == "__main__":
    main()