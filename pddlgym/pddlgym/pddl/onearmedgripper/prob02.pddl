(define (problem strips-gripper-x-2)
   (:domain onearmedgripper)
   (:objects 
        rooma - room
        roomb - room
        ball6 - ball
        ball5 - ball
        ball4 - ball
        ball3 - ball
        ball2 - ball
        ball1 - ball
        left - gripper
   )
   (:init (room rooma)
          (room roomb)
          (ball ball6)
          (ball ball5)
          (ball ball4)
          (ball ball3)
          (ball ball2)
          (ball ball1)
          (at-robby rooma)
          (free left)
          (at ball6 rooma)
          (at ball5 rooma)
          (at ball4 rooma)
          (at ball3 rooma)
          (at ball2 rooma)
          (at ball1 rooma)
          (gripper left))
   (:goal (and (at ball6 roomb)
               (at ball5 roomb)
               (at ball4 roomb)
               (at ball3 roomb)
               (at ball2 roomb)
               (at ball1 roomb))))