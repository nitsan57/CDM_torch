
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
	b25 - block
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
	(clear b14)
	(clear b16)
	(clear b17)
	(clear b18)
	(clear b20)
	(clear b21)
	(clear b23)
	(clear b24)
	(clear b2)
	(clear b3)
	(clear b5)
	(clear b7)
	(clear b8)
	(clear b9)
	(handempty )
	(on b0 b1)
	(on b12 b13)
	(on b14 b15)
	(on b18 b19)
	(on b21 b22)
	(on b24 b25)
	(on b3 b4)
	(on b5 b6)
	(on b9 b10)
	(ontable b10)
	(ontable b11)
	(ontable b13)
	(ontable b15)
	(ontable b16)
	(ontable b17)
	(ontable b19)
	(ontable b1)
	(ontable b20)
	(ontable b22)
	(ontable b23)
	(ontable b25)
	(ontable b2)
	(ontable b4)
	(ontable b6)
	(ontable b7)
	(ontable b8)
  )
  (:goal (and
	(on b7 b22)
	(on b22 b25)
	(ontable b25)
	(on b17 b11)
	(on b11 b3)
	(ontable b3)))
)
