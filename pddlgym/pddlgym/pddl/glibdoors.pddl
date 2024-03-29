(define (domain glibdoors)
  (:requirements :strips :typing)
  (:types location room key)
  (:predicates
     (at ?loc - location)
     (unlocked ?room - room)
     (locinroom ?loc - location ?room - room)
     (keyat ?key - key ?loc - location)
     (keyforroom ?key - key ?room - room)
     (moveto ?loc - location)
     (pick ?key - key)
  )

  ; (:actions moveto pick)

  (:action moveto
    :parameters (?sloc - location ?eloc - location ?eroom - room)
    :precondition (and (moveto ?eloc)
                       (at ?sloc)
                       (unlocked ?eroom)
                       (locinroom ?eloc ?eroom)
                  )
    :effect (and (not (at ?sloc))
                 (at ?eloc)
            )
  )

  (:action pick
    :parameters (?loc - location ?key - key ?room - room)
    :precondition (and (pick ?key)
                       (at ?loc)
                       (keyat ?key ?loc)
                       (keyforroom ?key ?room)
                  )
    :effect (and (not (keyat ?key ?loc))
                 (unlocked ?room)
            )
  )

)