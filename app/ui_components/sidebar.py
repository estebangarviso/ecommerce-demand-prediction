"""Componente de barra lateral."""

from typing import Dict, List, Optional, Tuple
import streamlit as st

from ..components import ChartBuilder, DataFrameBuilder
from ..services import PricingService
from ..state_manager import SessionStateManager


# Constantes de validación (mismo valores que en data_processing.py)
MIN_ROLLING_WINDOW = 2
MAX_ROLLING_WINDOW = 12
DEFAULT_ROLLING_WINDOWS = [3, 6]


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
        """Renderiza el sidebar y retorna los valores del formulario."""
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
            form_data["rolling_windows"],
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
        # Rolling Windows Configuration (FUERA del form para ser reactivo)
        rolling_windows, validation_msg = self._render_rolling_windows_config()

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
                "Calcular Demanda",
                type="primary",
                icon=":material/rocket_launch:",
                disabled=validation_msg is not None,
            )

            # Obtener configuración actual del modelo
            current_model_windows = SessionStateManager.get_current_rolling_windows()

            # Advertencia si rolling_windows no coincide con el modelo
            if rolling_windows != current_model_windows:
                st.warning(
                    f"**🔄 Reentrenamiento requerido**\n\n"
                    f"El modelo actual soporta ventanas `{current_model_windows}`. "
                    f"Configuraste `{rolling_windows}`.\n\n"
                    f"**Al presionar 'Calcular Demanda'**, se enviará una solicitud a la API "
                    f"para reentrenar el modelo automáticamente con las nuevas ventanas. "
                    f"Este proceso puede tomar varios minutos.\n\n"
                    f"O puedes cambiar a 'Usar modelo actual {current_model_windows}' arriba.",
                    icon=":material/autorenew:",
                )

        return {
            "shop_cluster": shop_cluster,
            "item_price": item_price,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_windows": rolling_windows,
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

    def _render_rolling_windows_config(self) -> Tuple[List[int], Optional[str]]:
        """Renderiza la sección de configuración de ventanas rolling con validación.

        Returns:
            Tuple con (rolling_windows, mensaje_validacion)
            - rolling_windows: Lista de enteros con las ventanas configuradas
            - mensaje_validacion: None si es válido, string con error si no lo es
        """
        # Obtener configuración actual del modelo desde el estado de sesión
        current_model_windows = SessionStateManager.get_current_rolling_windows()

        with st.expander(
            ":grey[:material/settings:] Configuración Avanzada: Ventanas Rolling", expanded=False
        ):
            st.markdown(
                """
            Las **ventanas rolling** calculan promedios móviles de las ventas históricas.
            Por ejemplo, `rolling_mean_3` es el promedio de los últimos 3 meses.
            """
            )

            # Mostrar configuración actual del modelo
            st.info(
                f"**Configuración actual del modelo**: `{current_model_windows}`",
                icon=":material/model_training:",
            )
            # Método de entrada
            config_method = st.radio(
                "Selecciona configuración:",
                options=["current", "preset", "custom"],
                format_func=lambda x: {
                    "current": f":green[:material/check_circle:] Usar modelo actual {current_model_windows}",
                    "preset": ":gray[:material/autorenew:] Reentrenar con preset",
                    "custom": ":gray[:material/autorenew:] Reentrenar con configuración personalizada",
                }[x],
                horizontal=False,
                help="Cambiar rolling windows requiere reentrenar el modelo con las nuevas configuraciones.",
            )

            if config_method == "current":
                rolling_windows = current_model_windows
                st.success(
                    "Usando configuración actual. No requiere reentrenamiento.",
                    icon=":material/check_circle:",
                )

            elif config_method == "preset":
                preset_map = {
                    "default": DEFAULT_ROLLING_WINDOWS,
                    "short": [2, 4],
                    "long": [6, 12],
                }

                # Filtrar opciones que coinciden con current_model_windows
                available_presets = {
                    k: v for k, v in preset_map.items() if v != current_model_windows
                }

                if not available_presets:
                    st.warning(
                        "Todos los presets coinciden con el modelo actual. Usa 'custom' para otras configuraciones.",
                        icon=":material/info:",
                    )
                    rolling_windows = current_model_windows
                else:
                    preset_option = st.selectbox(
                        "Selecciona un preset:",
                        options=list(available_presets.keys()),
                        format_func=lambda x: {
                            "default": f"Predeterminado {available_presets[x]}",
                            "short": f"Corto plazo {available_presets[x]}",
                            "long": f"Largo plazo {available_presets[x]}",
                        }[x],
                        help="Presets optimizados para diferentes horizontes temporales.",
                    )
                    rolling_windows = available_presets[preset_option]

                    st.warning(
                        f"**Se activará reentrenamiento** con ventanas `{rolling_windows}`\n\n"
                        "El modelo se reentrenará automáticamente con la nueva configuración. "
                        "Este proceso puede tomar varios minutos.",
                        icon=":material/autorenew:",
                    )

            else:  # custom
                st.info("Debes especificar **exactamente 2 ventanas**.", icon=":material/info:")

                col1, col2 = st.columns(2)
                with col1:
                    window1 = st.number_input(
                        "Primera ventana (meses):",
                        min_value=MIN_ROLLING_WINDOW,
                        max_value=MAX_ROLLING_WINDOW - 1,  # Dejar espacio para window2
                        value=2,
                        step=1,
                        help=f"Ventana corta: entre {MIN_ROLLING_WINDOW} y {MAX_ROLLING_WINDOW - 1} meses",
                    )
                with col2:
                    # La segunda ventana debe ser mayor que la primera
                    min_window2 = (
                        int(window1) + 1 if window1 < MAX_ROLLING_WINDOW else MAX_ROLLING_WINDOW
                    )
                    window2 = st.number_input(
                        "Segunda ventana (meses):",
                        min_value=min_window2,
                        max_value=MAX_ROLLING_WINDOW,
                        value=max(4, min_window2),  # Al menos 4 o min_window2
                        step=1,
                        help=f"Ventana larga: debe ser mayor que {int(window1)} meses",
                    )

                rolling_windows = [int(window1), int(window2)]

                # Validación
                validation_msg = self._validate_rolling_windows(rolling_windows)

                if validation_msg:
                    st.error(f"{validation_msg}", icon=":material/error:")
                    return [3, 6], validation_msg

                # Verificar si coincide con el modelo actual
                if rolling_windows != current_model_windows:
                    st.warning(
                        f"**Se activará reentrenamiento** con ventanas `{rolling_windows}`\n\n"
                        "El modelo se reentrenará automáticamente con la nueva configuración. "
                        "Este proceso puede tomar varios minutos.",
                        icon=":material/autorenew:",
                    )
                else:
                    st.success(
                        "Configuración coincide con el modelo actual. No requiere reentrenamiento.",
                        icon=":material/check_circle:",
                    )

            return rolling_windows, None

    def _validate_rolling_windows(self, windows: List[int]) -> Optional[str]:
        """Valida la configuración de ventanas rolling.

        Args:
            windows: Lista de enteros con las ventanas a validar (DEBE tener exactamente 2)

        Returns:
            None si es válido, string con mensaje de error si no lo es
        """
        if not windows:
            return "Debe especificar al menos una ventana rolling."

        # NUEVA VALIDACIÓN: Exactamente 2 ventanas
        if len(windows) != 2:
            return f"Debes especificar EXACTAMENTE 2 ventanas. Recibido: {len(windows)}"

        if len(windows) != len(set(windows)):
            return "Las ventanas no deben tener valores duplicados."

        # Validar orden correcto
        if windows[0] >= windows[1]:
            return f"La primera ventana ({windows[0]}) debe ser menor que la segunda ({windows[1]})"

        for w in windows:
            if not isinstance(w, int):
                return f"La ventana '{w}' no es un número entero."
            if w < MIN_ROLLING_WINDOW or w > MAX_ROLLING_WINDOW:
                return f"La ventana {w} está fuera del rango válido [{MIN_ROLLING_WINDOW}, {MAX_ROLLING_WINDOW}]."

        return None  # Válido
