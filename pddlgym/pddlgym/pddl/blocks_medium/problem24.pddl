
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
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
	(clear b3)
	(clear b8)
	(handempty)
	(on b0 b1)
	(on b10 b11)
	(on b1 b2)
	(on b3 b4)
	(on b4 b5)
	(on b5 b6)
	(on b6 b7)
	(on b8 b9)
	(on b9 b10)
	(ontable b11)
	(ontable b2)
	(ontable b7)
  )
  (:goal (and
	(on b1 b11)
	(on b11 b3)
	(ontable b3)
	(on b9 b10)
	(on b10 b2)
	(on b2 b6)
	(ontable b6)
	(on b8 b0)
	(on b0 b7)
	(on b7 b4)
	(on b4 b5)
	(ontable b5)))
)
