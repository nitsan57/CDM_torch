
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
	b12 - block
	b13 - block
	b14 - block
	b2 - block
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
	(clear b4)
	(clear b7)
	(handempty)
	(on b0 b1)
	(on b11 b12)
	(on b12 b13)
	(on b13 b14)
	(on b1 b2)
	(on b2 b3)
	(on b4 b5)
	(on b5 b6)
	(on b7 b8)
	(on b8 b9)
	(on b9 b10)
	(ontable b10)
	(ontable b14)
	(ontable b3)
	(ontable b6)
  )
  (:goal (and
	(on b2 b3)
	(on b3 b8)
	(ontable b8)
	(on b5 b6)
	(on b6 b0)
	(on b0 b10)
	(on b10 b13)
	(ontable b13)
	(on b14 b9)
	(on b9 b7)
	(ontable b7)))
)
