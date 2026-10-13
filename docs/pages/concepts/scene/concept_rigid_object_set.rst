Rigid Object Sets
=================

Use a ``RigidObjectSet`` to spawn **heterogeneous objects across parallel
environments** — different rigid objects per env, without changing the scene
or task definition.

Motivation
----------

A plain ``Object`` clones the same USD into every environment. To spawn a
different object per env without rewriting the environment definition, wrap the
candidates in a ``RigidObjectSet``. Arena assigns one member per environment
before spawn and uses that member's spawn configuration and bounding box for
placement.

**Example.** Suppose the task is to pick a fruit and place it in a bowl. With a
single ``Object``, every parallel environment tests the same fruit. With a
``RigidObjectSet`` of banana, orange, and lemon, each environment can test a
different fruit — same pick-and-place task, richer coverage in one run.

What It Is
----------

``RigidObjectSet`` is a subclass of ``Object`` that holds a list of rigid
members. To the scene and task it still looks like one asset — one name, one
prim path, and (optionally) one set of relations. Only the spawned object
differs across environments.

When the environment is built and ``num_envs`` is known, Arena assigns one
member to each environment. That choice is locked for the run, so the spawned
member and the geometry used by the placement solver stay aligned.

How members are chosen:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Behavior
   * - Ordered (default)
     - Environments cycle through the list: ``env_idx % len(objects)``.
   * - Random (``random_choice=True``)
     - Each environment picks independently. Pass a placement seed to make the
       choice reproducible.

Members must be rigid objects. Articulations, empty lists and other object
sets are rejected when you construct the set.

Each member spawns from a copy of its own spawn configuration
(``member.object_cfg.spawn``), taken when the set is built. Its scale, physics
properties, USD variant selections and materials stay as configured. Later
edits to the member do not change what the set spawns, but the set's placement
bounds still read the member objects, so configure each member before building
the set. Set-wide settings can be changed on the set's ``object_cfg.spawn``.
Per-environment assignment rebuilds the entries in
``object_cfg.spawn.assets_cfg`` from the member copies, so edits to those
entries are replaced.

A member may also be a primitive shape, for example an ``Object`` with
``spawner_cfg=CuboidCfg(...)`` and ``rigid_props`` set. A member built from
``spawner_cfg`` (for example a primitive shape) keeps its own
``activate_contact_sensors`` value, which defaults to ``False``, so set it to
``True`` when a task needs contact sensing. Members built from ``usd_path`` have
it enabled.

How to Use
----------

Build the fruit members, wrap them in a set, add the set to the scene, and pass
it to the task as the pick object:

.. tab-set::

   .. tab-item:: Python
      :selected:

      .. code-block:: python

         from isaaclab_arena.assets.object_set import RigidObjectSet
         from isaaclab_arena.relations.relations import On
         from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

         members = [
             asset_registry.get_asset_by_name("banana_ycb_robolab")(),
             asset_registry.get_asset_by_name("orange_01_fruits_veggies_robolab")(),
             asset_registry.get_asset_by_name("lemon_01_fruits_veggies_robolab")(),
         ]
         fruit_set = RigidObjectSet(name="fruit", objects=members)
         fruit_set.add_relation(On(table_reference))

         scene = Scene(assets=[background, table, bowl, fruit_set])
         task = PickAndPlaceTask(
             pick_up_object=fruit_set,
             destination_location=bowl,
             background_scene=background,
         )

   .. tab-item:: YAML

      In an environment-graph spec, declare the set under ``object_sets`` and
      reference its ``id`` from relations and the task — the same way you would
      reference a single object:

      .. code-block:: yaml

         objects:
         - id: bowl
           registry_name: bowl_ycb_robolab
           params: {}
         object_sets:
         - id: fruit
           members:
           - banana_ycb_robolab
           - orange_01_fruits_veggies_robolab
           - lemon_01_fruits_veggies_robolab
           random_choice: false
           params: {}
         relations:
         - kind: 'on'
           subject: fruit
           reference: maple_table
           params: {}
         task:
           composition: atomic
           subtasks:
           - kind: PickAndPlaceTask
             params:
               pick_up_object: fruit
               destination_location: bowl
               background_scene: maple_table

With ``--num_envs 3`` and ordered assignment, one environment gets the banana,
one the orange, and one the lemon. The task definition stays the same; only the
spawned fruit differs.

A few things to know:

- **Set ``num_envs`` first.** Arena needs the environment count before it can
  assign members — in runners this is ``--num_envs``. Use more than one to see
  different members side by side.
- **Tasks take the set directly.** Anything that accepts an ``Object`` works;
  Arena resolves the active member through the shared prim path.

For a placement-focused comparison of same-object vs set-based roles, see
:doc:`../object_placement/homogeneous_and_heterogeneous_placement`.

Limitations
-----------

- **Rigid only.** Members must be ``ObjectType.RIGID`` — no articulations.
- **Choice is fixed per run.** Layouts can change on reset; the assigned member
  cannot.
- **One shared prim path and rigid-body path.** Every environment uses
  ``{ENV_REGEX_NS}/<name>``, and Isaac Lab looks up the rigid body at one path
  below it for all environments. When members keep their rigid bodies at
  different paths, Arena writes a copy of each member's USD with the body moved
  to ``/rigid_body`` and caches it as ``rigid_<sha256>.usd``. This needs USD
  members, and their spawn configurations cannot name prims inside the asset:
  ``prim_physics``, schema fragments or target patterns in ``*_props``,
  ``visual_material_bindings``, ``physics_material_prim_path``, and a relative
  ``visual_material_path`` when ``visual_material`` is set, are rejected. The
  copy keeps materials and bakes in the USD variant selections. An authored
  translation or rotation on the asset's root is dropped, as when Isaac Lab
  spawns it.
- **Single-box geometry has a fallback.** ``get_bounding_box()`` returns one box
  from the tallest member; per-environment placement uses
  ``get_bounding_box_per_env()`` instead.
- **YAML members can't carry a ``usd_path``.** Graph YAML builds members with no
  constructor args, so SimReady assets that need a custom path can't be YAML set
  members. In Python you pass live instances, so SimReady members are fine.
- **Don't set an initial pose when placement owns it.** The builder supplies the
  set's creation and reset poses. Anchors are the exception — they stay fixed and
  still need a known pose.
- **Contact sensors use the shared rigid-body path.** A task's contact sensor
  sits on the rigid body at the path that all members share.
- **Not a fixed obstacle.** A set has no ``usd_path`` of its own, so a set
  without relations is not used as a collision obstacle when other objects are
  placed. In ``MESH`` collision mode, a set with more than one member uses its
  bounding box.
