
(define (problem meet-pass3) (:domain meet-pass)
  (:objects
        siding1 - default
	track1 - default
	track2 - default
	track3 - default
	track4 - default
	track5 - default
	train1 - default
	train2 - default
	train3 - default
  )
  (:goal (and
	(at train1 track5)))
  (:init 
	(at train1 track1)
	(at train2 track2)
	(at train3 track4)
	(clear siding1)
	(clear track3)
	(clear track5)
	(connected siding1 track3)
	(connected track1 track2)
	(connected track2 track1)
	(connected track2 track3)
	(connected track3 siding1)
	(connected track3 track2)
	(connected track3 track4)
	(connected track4 track3)
	(connected track4 track5)
	(connected track5 track4)
	(move siding1 siding1)
	(move siding1 track1)
	(move siding1 track2)
	(move siding1 track3)
	(move siding1 track4)
	(move siding1 track5)
	(move siding1 train1)
	(move siding1 train2)
	(move siding1 train3)
	(move track1 siding1)
	(move track1 track1)
	(move track1 track2)
	(move track1 track3)
	(move track1 track4)
	(move track1 track5)
	(move track1 train1)
	(move track1 train2)
	(move track1 train3)
	(move track2 siding1)
	(move track2 track1)
	(move track2 track2)
	(move track2 track3)
	(move track2 track4)
	(move track2 track5)
	(move track2 train1)
	(move track2 train2)
	(move track2 train3)
	(move track3 siding1)
	(move track3 track1)
	(move track3 track2)
	(move track3 track3)
	(move track3 track4)
	(move track3 track5)
	(move track3 train1)
	(move track3 train2)
	(move track3 train3)
	(move track4 siding1)
	(move track4 track1)
	(move track4 track2)
	(move track4 track3)
	(move track4 track4)
	(move track4 track5)
	(move track4 train1)
	(move track4 train2)
	(move track4 train3)
	(move track5 siding1)
	(move track5 track1)
	(move track5 track2)
	(move track5 track3)
	(move track5 track4)
	(move track5 track5)
	(move track5 train1)
	(move track5 train2)
	(move track5 train3)
	(move train1 siding1)
	(move train1 track1)
	(move train1 track2)
	(move train1 track3)
	(move train1 track4)
	(move train1 track5)
	(move train1 train1)
	(move train1 train2)
	(move train1 train3)
	(move train2 siding1)
	(move train2 track1)
	(move train2 track2)
	(move train2 track3)
	(move train2 track4)
	(move train2 track5)
	(move train2 train1)
	(move train2 train2)
	(move train2 train3)
	(move train3 siding1)
	(move train3 track1)
	(move train3 track2)
	(move train3 track3)
	(move train3 track4)
	(move train3 track5)
	(move train3 train1)
	(move train3 train2)
	(move train3 train3)
))
        