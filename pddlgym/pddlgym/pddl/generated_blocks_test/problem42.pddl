
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b2 - block
	b3 - block
	b4 - block
	b5 - block
	b6 - block
  )
  (:init 
	(clear b0)
	(clear b2)
	(clear b4)
	(clear b5)
	(handempty )
	(on b0 b1)
	(on b2 b3)
	(on b5 b6)
	(ontable b1)
	(ontable b3)
	(ontable b4)
	(ontable b6)
  )
  (:goal (and
	(on b0 b3)
	(ontable b3)
	(on b4 b1)
	(on b1 b2)
	(ontable b2)
	(on b6 b5)
	(ontable b5)))
)
