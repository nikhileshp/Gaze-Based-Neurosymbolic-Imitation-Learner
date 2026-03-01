(define (domain local_mined)
  (:requirements :strips)
  (:predicates
    (above_water ?x ?y)
    (below_water ?x ?y)
    (close_by_enemy ?x ?y)
    (deeper_than_enemy ?x ?y)
    (facing_left ?x ?y)
    (facing_right ?x ?y)
    (higher_than_enemy ?x ?y)
    (is_collected_diver ?x ?y)
    (left_of_enemy ?x ?y)
    (oxygen_critical ?x ?y)
    (oxygen_low ?x ?y)
    (right_of_enemy ?x ?y)
    (same_depth_enemy ?x ?y)
    (visible_diver ?x ?y)
    (visible_enemy ?x ?y)
    (visible_missile ?x ?y)
  )

  (:action player_navigation
    :parameters (?focus)
    :precondition (and )
    :effect (and
    )
  )

  (:action engaging_submarine
    :parameters (?focus)
    :precondition (and visible_enemy(?focus) close_by_enemy(obj0,?focus) below_water(obj0))
    :effect (and
      not(right_of_enemy(obj0,?focus))
      not(same_depth_enemy(obj0,?focus))
      not(visible_enemy(?focus))
      not(left_of_enemy(obj0,?focus))
      not(deeper_than_enemy(obj0,?focus))
      not(close_by_enemy(obj0,?focus))
    )
  )

  (:action tracking_enemy
    :parameters (?focus)
    :precondition (and below_water(obj0) visible_enemy(?focus))
    :effect (and
      not(same_depth_enemy(obj0,?focus))
      not(visible_enemy(?focus))
      not(close_by_enemy(obj0,?focus))
    )
  )

  (:action rescuing_diver
    :parameters (?focus)
    :precondition (and below_water(obj0) visible_diver(?focus))
    :effect (and
      not(visible_diver(?focus))
    )
  )

  (:action saving_diver
    :parameters (?focus)
    :precondition (and above_water(obj0) is_collected_diver(?focus))
    :effect (and
      not(is_collected_diver(?focus))
    )
  )

  (:action surfacing
    :parameters (?focus)
    :precondition (and )
    :effect (and
      above_water(obj0)
      not(below_water(obj0))
    )
  )

  (:action checking_oxygen
    :parameters (?focus)
    :precondition (and )
    :effect (and
      not(oxygen_low(?focus))
      not(oxygen_critical(?focus))
    )
  )

  (:action attacking_shark
    :parameters (?focus)
    :precondition (and below_water(obj0) visible_enemy(?focus))
    :effect (and
      facing_right(obj0)
      not(higher_than_enemy(obj0,?focus))
      not(facing_left(obj0))
      not(visible_enemy(?focus))
      not(left_of_enemy(obj0,?focus))
    )
  )

  (:action dodging_missile
    :parameters (?focus)
    :precondition (and below_water(obj0) visible_missile(?focus))
    :effect (and
      not(visible_missile(?focus))
    )
  )

)
