
(define (problem manygripper) (:domain gripper-strips)
  (:objects
        ball0
	ball1
	ball10
	ball11
	ball12
	ball13
	ball14
	ball15
	ball16
	ball17
	ball18
	ball19
	ball2
	ball20
	ball21
	ball22
	ball23
	ball24
	ball25
	ball26
	ball27
	ball28
	ball29
	ball3
	ball4
	ball5
	ball6
	ball7
	ball8
	ball9
	gripper0
	gripper1
	room0
	room1
	room10
	room2
	room3
	room4
	room5
	room6
	room7
	room8
	room9
  )
  (:init 
	(at ball0 room10)
	(at ball10 room6)
	(at ball11 room4)
	(at ball12 room7)
	(at ball13 room5)
	(at ball14 room5)
	(at ball15 room3)
	(at ball16 room8)
	(at ball17 room7)
	(at ball18 room10)
	(at ball19 room1)
	(at ball1 room1)
	(at ball20 room7)
	(at ball21 room5)
	(at ball22 room10)
	(at ball23 room4)
	(at ball24 room0)
	(at ball25 room5)
	(at ball26 room10)
	(at ball27 room7)
	(at ball28 room8)
	(at ball29 room1)
	(at ball2 room8)
	(at ball3 room8)
	(at ball4 room3)
	(at ball5 room0)
	(at ball6 room9)
	(at ball7 room2)
	(at ball8 room9)
	(at ball9 room3)
	(at-robby room0)
	(ball ball0)
	(ball ball10)
	(ball ball11)
	(ball ball12)
	(ball ball13)
	(ball ball14)
	(ball ball15)
	(ball ball16)
	(ball ball17)
	(ball ball18)
	(ball ball19)
	(ball ball1)
	(ball ball20)
	(ball ball21)
	(ball ball22)
	(ball ball23)
	(ball ball24)
	(ball ball25)
	(ball ball26)
	(ball ball27)
	(ball ball28)
	(ball ball29)
	(ball ball2)
	(ball ball3)
	(ball ball4)
	(ball ball5)
	(ball ball6)
	(ball ball7)
	(ball ball8)
	(ball ball9)
	(free gripper0)
	(free gripper1)
	(gripper gripper0)
	(gripper gripper1)
	(room room0)
	(room room10)
	(room room1)
	(room room2)
	(room room3)
	(room room4)
	(room room5)
	(room room6)
	(room room7)
	(room room8)
	(room room9)
  )
  (:goal (and
	(at ball12 room7)
	(at ball17 room5)
	(at ball5 room3)
	(at ball14 room7)))
)
