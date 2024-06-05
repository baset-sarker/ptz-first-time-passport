from Focuser import Focuser
from AutoFocus import AutoFocus
from JetsonCamera import Camera

def test(focuser):
    motor_step  = 5
    focus_step  = 100
    zoom_step   = 100
    print(focuser.get(Focuser.OPT_MOTOR_Y))
    print(focuser.get(Focuser.OPT_MOTOR_X))
    #focuser.set(Focuser.OPT_MOTOR_Y,focuser.get(Focuser.OPT_MOTOR_Y) + motor_step)
    #focuser.reset(Focuser.OPT_FOCUS)
    # while focuser.get(Focuser.OPT_FOCUS) < 18000:
    #     focuser.set(Focuser.OPT_FOCUS,focuser.get(Focuser.OPT_FOCUS) + 50)
    # focuser.set(Focuser.OPT_FOCUS,0)
    # focuser.set(Focuser.OPT_FOCUS,10000)
pass

def init_focuser(focuser):
    focuser.set(Focuser.OPT_MOTOR_Y,175)
    focuser.set(Focuser.OPT_MOTOR_X,90)
    # focuser.set(Focuser.OPT_MOTOR_Y,0)
    # focuser.set(Focuser.OPT_MOTOR_X,0)


if __name__ == "__main__":
    focuser = Focuser(1)
    init_focuser(focuser)
    # camera = Camera()
    # init_focuser(focuser)
    # auto_focus = AutoFocus(focuser,camera)
    # auto_focus.debug = True
    # max_index,max_value = auto_focus.startFocus()
    # #test(focuser)
    # camera.close()