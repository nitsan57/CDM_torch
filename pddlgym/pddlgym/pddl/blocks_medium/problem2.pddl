
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
	(clear b12)
	(clear b16)
	(clear b20)
	(clear b3)
	(clear b8)
	(handempty)
	(on b0 b1)
	(on b10 b11)
	(on b12 b13)
	(on b13 b14)
	(on b14 b15)
	(on b16 b17)
	(on b17 b18)
	(on b18 b19)
	(on b1 b2)
	(on b20 b21)
	(on b21 b22)
	(on b3 b4)
	(on b4 b5)
	(on b5 b6)
	(on b6 b7)
	(on b8 b9)
	(on b9 b10)
	(ontable b11)
	(ontable b15)
	(ontable b19)
	(ontable b22)
	(ontable b2)
	(ontable b7)
  )
  (:goal (and
	(on b16 b1)
	(on b1 b6)
	(on b6 b21)
	(ontable b21)
	(on b17 b18)
	(on b18 b15)
	(on b15 b14)
	(ontable b14)
	(on b4 b22)
	(on b22 b11)
	(ontable b11)
	(on b19 b10)
	(on b10 b3)
	(on b3 b20)
	(ontable b20)))
)
