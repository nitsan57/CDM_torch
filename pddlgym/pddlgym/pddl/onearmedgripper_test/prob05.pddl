(define (problem strips-gripper-x-5)
   (:domain onearmedgripper)
   (:objects
        rooma - room
        roomb - room 
        ball12 - ball
        ball11 - ball 
        ball10 - ball 
        ball9 - ball 
        ball8 - ball 
        ball7 - ball 
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
          (ball ball12)
          (ball ball11)
          (ball ball10)
          (ball ball9)
          (ball ball8)
          (ball ball7)
          (ball ball6)
          (ball ball5)
          (ball ball4)
          (ball ball3)
          (ball ball2)
          (ball ball1)
          (at-robby rooma)
          (free left)
          (at ball12 rooma)
          (at ball11 rooma)
          (at ball10 rooma)
          (at ball9 rooma)
          (at ball8 rooma)
          (at ball7 rooma)
          (at ball6 rooma)
          (at ball5 rooma)
          (at ball4 rooma)
          (at ball3 rooma)
          (at ball2 rooma)
          (at ball1 rooma)
          (gripper left))
   (:goal (and (at ball12 roomb)
               (at ball11 roomb)
               (at ball10 roomb)
               (at ball9 roomb)
               (at ball8 roomb)
               (at ball7 roomb)
               (at ball6 roomb)
               (at ball5 roomb)
               (at ball4 roomb)
               (at ball3 roomb)
               (at ball2 roomb)
               (at ball1 roomb))))