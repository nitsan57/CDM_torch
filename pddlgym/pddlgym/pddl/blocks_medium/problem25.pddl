
(define (problem generatedblocks) (:domain blocks)
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
	(clear b13)
	(clear b17)
	(clear b21)
	(clear b5)
	(clear b9)
	(handempty)
	(on b0 b1)
	(on b10 b11)
	(on b11 b12)
	(on b13 b14)
	(on b14 b15)
	(on b15 b16)
	(on b17 b18)
	(on b18 b19)
	(on b19 b20)
	(on b1 b2)
	(on b21 b22)
	(on b22 b23)
	(on b23 b24)
	(on b24 b25)
	(on b2 b3)
	(on b3 b4)
	(on b5 b6)
	(on b6 b7)
	(on b7 b8)
	(on b9 b10)
	(ontable b12)
	(ontable b16)
	(ontable b20)
	(ontable b25)
	(ontable b4)
	(ontable b8)
  )
  (:goal (and
	(on b20 b7)
	(on b7 b14)
	(ontable b14)
	(on b21 b18)
	(on b18 b4)
	(on b4 b19)
	(ontable b19)
	(on b25 b15)
	(on b15 b24)
	(ontable b24)))
)
