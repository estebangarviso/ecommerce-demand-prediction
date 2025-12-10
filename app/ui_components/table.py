"""Componente de tabla mejorada utilizando Great Tables."""

from typing import Any, Dict, Literal, cast
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.great_tables import great_tables
from great_tables import GT, md
from pandas import DataFrame


class Table:
    """Componente de tabla mejorada utilizando Great Tables."""

    def __init__(
        self,
        rowname_col: str | None = None,
        groupname_col: str | None = None,
        auto_align: bool = True,
        id: str | None = None,
        width: int | Literal["stretch", "content"] = "stretch",
    ):
        """Inicializa el componente de tabla."""
        self.id = id
        self.locale = st.session_state.get("locale", "es_CL")
        self.rowname_col = rowname_col
        self.groupname_col = groupname_col
        self.auto_align = auto_align
        self.width = width
        self.table = None

    @staticmethod
    def create_bar(prop_fill: float, max_width: int, height: int) -> str:
        """Crea una barra de progreso en HTML."""
        width = round(max_width * prop_fill, 2)
        px_width = f"{width}px"
        percentage = round(prop_fill * 100, 2)
        return f"""\
        <div style="width: {max_width}px; background-color: lightgrey; position: relative; height: {height}px;">\
            <div style="height:{height}px;width:{px_width};background-color:green;position:absolute;top:0;left:0;"></div>\
            <div style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;text-shadow:1px 1px 2px rgba(0,0,0,0.5);">{percentage}%</div>\
        </div>\
        """

    def get_table(self, data: DataFrame) -> "GT":
        """
        Configura la tabla a renderizar.
        Ejemplos: https://posit-dev.github.io/great-tables/examples/
        """
        return GT(
            data=data,
            id=self.id,
            locale=self.locale,
            rowname_col=self.rowname_col,
            groupname_col=self.groupname_col,
            auto_align=self.auto_align,
        )

    def render(self, table: "GT") -> None:
        """Renderiza una tabla mejorada en Streamlit."""
        if self.width == "stretch":
            table = table.tab_options(container_width="100%", table_width="100%")
        elif self.width == "content":
            # Do nothing -> uses content as default.
            pass
        else:
            table = table.tab_options(
                container_width=f"{self.width}px", table_width=f"{self.width}px"
            )
        # https://posit-dev.github.io/great-tables/reference/GT.tab_options.html#great_tables.GT.tab_options
        st.write(table.as_raw_html(), unsafe_allow_html=True)
