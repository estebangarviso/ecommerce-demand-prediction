"""Componente de encabezado."""

import streamlit as st


class Header:
    """Componente de encabezado."""

    @staticmethod
    def render() -> None:
        """Renderiza el encabezado de la aplicación."""
        col_h1, col_h2 = st.columns([0.08, 0.92])
        with col_h1:
            st.markdown("# :material/query_stats:")
        with col_h2:
            st.title("Sistema Predictivo de Demanda")
            st.markdown("Ensamble Stacking & Deep Learning • API REST")
        st.divider()
