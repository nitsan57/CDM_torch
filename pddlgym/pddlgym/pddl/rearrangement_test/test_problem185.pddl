(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	monkey-0 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-0-3 - static
	loc-0-4 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-1-3 - static
	loc-1-4 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-2-3 - static
	loc-2-4 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
	loc-3-3 - static
	loc-3-4 - static
	loc-4-0 - static
	loc-4-1 - static
	loc-4-2 - static
	loc-4-3 - static
	loc-4-4 - static
    )

    (:init
    
	(ismonkey monkey-0)
	(isrobot robot)
	(at monkey-0 loc-4-3)
	(at robot loc-2-3)
	(handsfree robot)

    ; action literals
    
	(pick monkey-0)
	(place monkey-0)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-0-3)
	(moveto loc-0-4)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-1-3)
	(moveto loc-1-4)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
	(moveto loc-2-3)
	(moveto loc-2-4)
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
	(moveto loc-3-3)
	(moveto loc-3-4)
	(moveto loc-4-0)
	(moveto loc-4-1)
	(moveto loc-4-2)
	(moveto loc-4-3)
	(moveto loc-4-4)
    )

    (:goal (and  (at monkey-0 loc-0-1) ))
)
    