
(define (problem easynewspaper) (:domain easynewspapers)
  (:objects
        loc-0 - loc
	loc-1 - loc
	loc-2 - loc
	paper-0 - paper
	paper-1 - paper
	paper-2 - paper
	paper-3 - paper
	paper-4 - paper
  )
  (:init 
	(at loc-0)
	(ishomebase loc-0)
	(unpacked paper-0)
	(unpacked paper-1)
	(unpacked paper-2)
	(unpacked paper-3)
	(unpacked paper-4)
	(wantspaper loc-1)
	(wantspaper loc-2)
  )
  (:goal (and
	(satisfied loc-1)
	(satisfied loc-2)))
)
