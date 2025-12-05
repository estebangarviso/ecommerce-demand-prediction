"""Componente de barra lateral."""

from typing import Dict
import streamlit as st

from ..components import ChartBuilder, DataFrameBuilder
from ..services import PricingService
from ..state_manager import SessionStateManager


class Sidebar:
    """Componente de barra lateral."""

    def __init__(
        self,
        category_map: Dict[int, str],
        cluster_map: Dict[int, str],
        pricing_service: PricingService,
    ):
        """
        Inicializa el sidebar.

        Args:
            category_map: Mapa de categorías
            cluster_map: Mapa de clusters
            pricing_service: Servicio de precios
        """
        self.category_map = category_map
        self.cluster_map = cluster_map
        self.pricing_service = pricing_service
        self.chart_builder = ChartBuilder()
        self.df_builder = DataFrameBuilder()

    def render(self) -> tuple:
        """
        Renderiza el sidebar y retorna los valores del formulario.

        Returns:
            Tupla con (item_category_id, shop_cluster, item_price, lag_1, lag_2, lag_3, predict_btn)
        """
        with st.sidebar:
            st.header("Parámetros", divider="blue")
            st.info("Configure el escenario de venta.", icon=":material/tune:")

            # Categoría
            item_category_id = self._render_category_selector()

            # Formulario
            form_data = self._render_prediction_form()

            # Gráfico de tendencia
            self._render_trend_chart(form_data["lag_3"], form_data["lag_2"], form_data["lag_1"])

        return (
            item_category_id,
            form_data["shop_cluster"],
            form_data["item_price"],
            form_data["lag_1"],
            form_data["lag_2"],
            form_data["lag_3"],
            form_data["predict_btn"],
        )

    def _render_category_selector(self) -> int:
        """Renderiza el selector de categoría."""
        st.subheader(":material/category: Producto")

        def on_category_change():
            cat_id = SessionStateManager.get_value(SessionStateManager.SELECTED_CATEGORY)
            self.pricing_service.update_price_for_category(cat_id)

        item_category_id = st.selectbox(
            "Categoría:",
            options=list(self.category_map.keys()),
            format_func=lambda x: f"{x} - {self.category_map[x]}",
            label_visibility="collapsed",
            key=SessionStateManager.SELECTED_CATEGORY,
            on_change=on_category_change,
            help="Al cambiar la categoría, se sugerirá un precio promedio histórico.",
        )

        # Actualizar precio si la categoría ha cambiado
        last_category = SessionStateManager.get_value(SessionStateManager.LAST_CATEGORY)
        if last_category != item_category_id:
            self.pricing_service.update_price_for_category(item_category_id)
            SessionStateManager.set_value(SessionStateManager.LAST_CATEGORY, item_category_id)

        return item_category_id

    def _render_prediction_form(self) -> Dict:
        """Renderiza el formulario de predicción."""
        with st.form("prediction_form"):
            # Cluster de tienda
            st.subheader(":material/store: Perfil de Tienda")
            shop_cluster = st.selectbox(
                "Seleccione el perfil:",
                options=list(self.cluster_map.keys()),
                format_func=lambda x: self.cluster_map[x],
                label_visibility="collapsed",
                # Tooltip explicativo sobre K-Means
                help="Segmentación automática basada en el volumen histórico de ventas (Modelo K-Means). Ayuda al sistema a distinguir entre tiendas pequeñas, medianas y grandes.",
            )

            # Precio
            st.subheader(":material/sell: Precio")
            _price, price_min, price_max = self.pricing_service.get_current_price_range()
            item_price = st.slider(
                "Precio Unitario ($)",
                price_min,
                price_max,
                key=SessionStateManager.PRICE_SLIDER,
                step=1.0 if price_max < 500 else 10.0,
                help="Precio de venta unitario. Esta variable tiene una correlación inversa con la demanda (a mayor precio, suele bajar la venta).",
            )

            # Ventas históricas
            st.divider()
            st.subheader(":material/history: Ventas Históricas")
            col1, col2, col3 = st.columns(3)
            with col1:
                lag_1 = st.number_input(
                    "Mes t-1",
                    0,
                    1000,
                    5,
                    help="Ventas del mes inmediatamente anterior. Es el predictor más fuerte del modelo (Inercia).",
                )
            with col2:
                lag_2 = st.number_input(
                    "Mes t-2",
                    0,
                    1000,
                    4,
                    help="Ventas de hace 2 meses. Permite capturar la tendencia a corto plazo.",
                )
            with col3:
                lag_3 = st.number_input(
                    "Mes t-3",
                    0,
                    1000,
                    4,
                    help="Ventas de hace 3 meses. Ayuda a suavizar el ruido de variaciones recientes.",
                )

            predict_btn = st.form_submit_button(
                "Calcular Demanda", type="primary", icon=":material/rocket_launch:"
            )

        return {
            "shop_cluster": shop_cluster,
            "item_price": item_price,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "predict_btn": predict_btn,
        }

    def _render_trend_chart(self, lag_3: int, lag_2: int, lag_1: int) -> None:
        """Renderiza el gráfico de tendencia de entrada."""
        st.markdown("---")
        st.markdown("###### :material/input: Tendencia de Entrada")
        st.caption("Evolución de ventas previas ingresadas.")

        input_trend_df = self.df_builder.create_trend_dataframe(lag_3, lag_2, lag_1)
        fig_input = self.chart_builder.create_bar_chart(input_trend_df, "Mes", "Ventas")
        st.plotly_chart(fig_input, width="stretch", config={"displayModeBar": False})
