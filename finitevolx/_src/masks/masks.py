import typing as tp
from jaxtyping import Array
import jax.numpy as jnp

from finitevolx._src.interp.interp import avg_pool


class VariableMask(tp.NamedTuple):
    values: Array
    not_values: Array
    distbound1: Array
    irrbound_xids: Array
    irrbound_yids: Array

class VelocityMask(tp.NamedTuple):
    values: Array
    not_values: Array
    distbound1: Array
    distbound2: Array
    distbound2plus: Array
    distbound3plus: Array

class TracerMask(tp.NamedTuple):
    values: Array
    not_values: Array
    values_interior: Array
    distbound1: Array


class MaskDGrid(tp.NamedTuple):
    q: TracerMask
    u: VelocityMask
    v: VelocityMask
    psi: VariableMask

    @classmethod
    def init_mask(cls, mask: Array, variable: str = "q"):

        mtype = mask.dtype

        if variable == "q":
            q, u, v, psi = init_masks_dgrid_from_q(mask)
        elif variable == "psi":
            q, u, v, psi = init_masks_dgrid_from_psi(mask)
        else:
            raise ValueError(f"Unrecognized variable: {variable}")
        not_q = jnp.logical_not(q.astype(bool))
        not_u = jnp.logical_not(u.astype(bool))
        not_v = jnp.logical_not(v.astype(bool))
        not_psi = jnp.logical_not(psi.astype(bool))

        # VARIABLE
        psi_irrbound_xids = jnp.logical_and(
            not_psi[1:-1, 1:-1],
            avg_pool(psi, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)) > 1 / 18
        )
        psi_irrbound_xids = jnp.where(psi_irrbound_xids)

        psi_distbound1 = jnp.logical_and(
            avg_pool(psi.astype(mtype), (3, 3), stride=(1, 1), padding=(1, 1)) < 17 / 18,
            psi
        )

        # TRACER
        q_distbound1 = jnp.logical_and(
            avg_pool(q, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) < 17 / 18,
            q
        )
        q_interior = jnp.logical_and(jnp.logical_not(psi_distbound1), psi)

        u_distbound1 = jnp.logical_and(
            avg_pool(u, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)) < 5 / 6,
            u
        )
        v_distbound1 = jnp.logical_and(
            avg_pool(v, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)) < 5 / 6,
            v
        )

        u_distbound2plus = jnp.logical_and(
            jnp.logical_not(u_distbound1),
            u
        )
        v_distbound2plus = jnp.logical_and(
            jnp.logical_not(v_distbound1),
            v
        )

        u_distbound2 = jnp.logical_and(
            avg_pool(u, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)) < 9 / 10,
            u_distbound2plus
        )
        v_distbound2 = jnp.logical_and(
            avg_pool(v, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)) < 9 / 10,
            v_distbound2plus
        )

        u_distbound3plus = jnp.logical_and(
            jnp.logical_not(u_distbound2),
            u_distbound2plus
        )
        v_distbound3plus = jnp.logical_and(
            jnp.logical_not(v_distbound2),
            v_distbound2plus
        )

        # create variable mask
        psi = VariableMask(
            values=psi.astype(mtype),
            not_values=not_psi.astype(mtype),
            distbound1=psi_distbound1.astype(mtype),
            irrbound_xids=psi_irrbound_xids[0].astype(jnp.int32),
            irrbound_yids=psi_irrbound_xids[1].astype(jnp.int32),
        )

        # create tracer mask
        q = TracerMask(
            values=q.astype(mtype),
            not_values=not_q,
            distbound1=q_distbound1.astype(mtype),
            values_interior=q_interior.astype(mtype),
        )

        # create u velocity mask
        u = VelocityMask(
            values=u.astype(mtype),
            not_values=not_u.astype(mtype),
            distbound1=u_distbound1.astype(mtype),
            distbound2=u_distbound2.astype(mtype),
            distbound2plus=u_distbound2plus.astype(mtype),
            distbound3plus=u_distbound3plus.astype(mtype),
        )

        # create v velocity mask
        v = VelocityMask(
            values=v.astype(mtype),
            not_values=not_v,
            distbound1=v_distbound1.astype(mtype),
            distbound2plus=v_distbound2plus.astype(mtype),
            distbound2=v_distbound2.astype(mtype),
            distbound3plus=v_distbound3plus.astype(mtype),
        )

        return cls(
            psi=psi, u=u, v=v, q=q
        )

class MaskCGrid(tp.NamedTuple):
    q: TracerMask
    u: VelocityMask
    v: VelocityMask
    psi: VariableMask

    @classmethod
    def init_mask(cls, mask: Array, variable: str = "q"):

        mtype = mask.dtype

        if variable == "q":
            q, u, v, psi = init_masks_cgrid_from_q(mask)
        elif variable == "psi":
            q, u, v, psi = init_masks_cgrid_from_psi(mask)
        else:
            raise ValueError(f"Unrecognized variable: {variable}")
        not_q = jnp.logical_not(q.astype(bool))
        not_u = jnp.logical_not(u.astype(bool))
        not_v = jnp.logical_not(v.astype(bool))
        not_psi = jnp.logical_not(psi.astype(bool))

        # VARIABLE
        psi_irrbound_xids = jnp.logical_and(
            not_psi[1:-1, 1:-1],
            avg_pool(psi, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)) > 1 / 18
        )
        psi_irrbound_xids = jnp.where(psi_irrbound_xids)

        psi_distbound1 = jnp.logical_and(
            avg_pool(psi.astype(mtype), (3, 3), stride=(1, 1), padding=(1, 1)) < 17 / 18,
            psi
        )

        # TRACER
        q_distbound1 = jnp.logical_and(
            avg_pool(q, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) < 17 / 18,
            q
        )
        q_interior = jnp.logical_and(jnp.logical_not(psi_distbound1), psi)

        u_distbound1 = jnp.logical_and(
            avg_pool(u, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)) < 5 / 6,
            u
        )
        v_distbound1 = jnp.logical_and(
            avg_pool(v, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)) < 5 / 6,
            v
        )

        u_distbound2plus = jnp.logical_and(
            jnp.logical_not(u_distbound1),
            u
        )
        v_distbound2plus = jnp.logical_and(
            jnp.logical_not(v_distbound1),
            v
        )

        u_distbound2 = jnp.logical_and(
            avg_pool(u, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)) < 9 / 10,
            u_distbound2plus
        )
        v_distbound2 = jnp.logical_and(
            avg_pool(v, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)) < 9 / 10,
            v_distbound2plus
        )

        u_distbound3plus = jnp.logical_and(
            jnp.logical_not(u_distbound2),
            u_distbound2plus
        )
        v_distbound3plus = jnp.logical_and(
            jnp.logical_not(v_distbound2),
            v_distbound2plus
        )

        # create variable mask
        psi = VariableMask(
            values=psi.astype(mtype),
            not_values=not_psi.astype(mtype),
            distbound1=psi_distbound1.astype(mtype),
            irrbound_xids=psi_irrbound_xids[0].astype(jnp.int32),
            irrbound_yids=psi_irrbound_xids[1].astype(jnp.int32),
        )

        # create tracer mask
        q = TracerMask(
            values=q.astype(mtype),
            not_values=not_q,
            distbound1=q_distbound1.astype(mtype),
            values_interior=q_interior.astype(mtype),
        )

        # create u velocity mask
        u = VelocityMask(
            values=u.astype(mtype),
            not_values=not_u.astype(mtype),
            distbound1=u_distbound1.astype(mtype),
            distbound2=u_distbound2.astype(mtype),
            distbound2plus=u_distbound2plus.astype(mtype),
            distbound3plus=u_distbound3plus.astype(mtype),
        )

        # create v velocity mask
        v = VelocityMask(
            values=v.astype(mtype),
            not_values=not_v,
            distbound1=v_distbound1.astype(mtype),
            distbound2plus=v_distbound2plus.astype(mtype),
            distbound2=v_distbound2.astype(mtype),
            distbound3plus=v_distbound3plus.astype(mtype),
        )

        return cls(
            psi=psi, u=u, v=v, q=q
        )


def init_masks_dgrid_from_q(mask: Array):
    q = jnp.copy(mask)
    u = avg_pool(q, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4
    v = avg_pool(q, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4
    psi = avg_pool(q, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)) > 7 / 8
    return q, u, v, psi

def init_masks_cgrid_from_q(mask: Array):
    q = jnp.copy(mask)
    u = avg_pool(q, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4
    v = avg_pool(q, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4
    psi = avg_pool(q, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)) > 7 / 8
    return q, u, v, psi


def init_masks_dgrid_from_psi(mask: Array):
    psi = jnp.copy(mask)
    q = avg_pool(psi, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)) > 0.0
    u = avg_pool(q, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4
    v = avg_pool(q, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4

    return q, u, v, psi


def init_masks_cgrid_from_psi(mask: Array):
    psi = jnp.copy(mask)
    q = avg_pool(psi, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0)) > 0.0
    u = avg_pool(q, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4
    v = avg_pool(q, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4

    return q, u, v, psi