
(define (problem manyblockssmallpiles) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
	b12 - block
	b13 - block
	b14 - block
	b15 - block
	b16 - block
	b17 - block
	b18 - block
	b19 - block
	b2 - block
	b20 - block
	b21 - block
	b22 - block
	b23 - block
	b24 - block
	b3 - block
	b4 - block
	b5 - block
	b6 - block
	b7 - block
	b8 - block
	b9 - block
  )
  (:init 
	(clear b0)
	(clear b11)
	(clear b12)
	(clear b13)
	(clear b15)
	(clear b17)
	(clear b18)
	(clear b19)
	(clear b1)
	(clear b20)
	(clear b22)
	(clear b24)
	(clear b2)
	(clear b4)
	(clear b5)
	(clear b6)
	(clear b8)
	(clear b9)
	(handempty )
	(on b13 b14)
	(on b15 b16)
	(on b20 b21)
	(on b22 b23)
	(on b2 b3)
	(on b6 b7)
	(on b9 b10)
	(ontable b0)
	(ontable b10)
	(ontable b11)
	(ontable b12)
	(ontable b14)
	(ontable b16)
	(ontable b17)
	(ontable b18)
	(ontable b19)
	(ontable b1)
	(ontable b21)
	(ontable b23)
	(ontable b24)
	(ontable b3)
	(ontable b4)
	(ontable b5)
	(ontable b7)
	(ontable b8)
  )
  (:goal (and
	(on b10 b24)
	(ontable b24)))
)
