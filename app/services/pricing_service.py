"""Servicio para gestión de precios dinámicos."""

from typing import Dict, Optional, Tuple

from ..state_manager import SessionStateManager


class PricingService:
    """Servicio para gestión de precios dinámicos."""

    def __init__(
        self,
        cat_prices: Optional[Dict[int, float]],
        default_price: float,
        default_min: float,
        default_max: float,
        min_multiplier: float,
        max_multiplier: float,
    ):
        """Inicializa el servicio con precios por categoría y rangos por defecto."""
        self.cat_prices = cat_prices
        self.default_price = default_price
        self.default_min = default_min
        self.default_max = default_max
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def update_price_for_category(self, category_id: int) -> None:
        """Actualiza el precio basado en la categoría seleccionada."""
        if self.cat_prices and category_id in self.cat_prices:
            avg_price = float(self.cat_prices[category_id])
            SessionStateManager.update_price_range(
                avg_price, self.min_multiplier, self.max_multiplier
            )
        else:
            SessionStateManager.reset_price_range(
                self.default_price, self.default_min, self.default_max
            )

    def get_current_price_range(self) -> Tuple[float, float, float]:
        """Obtiene el rango actual de precios (actual, min, max)."""
        return (
            SessionStateManager.get_value(SessionStateManager.PRICE_SLIDER, self.default_price),
            SessionStateManager.get_value(SessionStateManager.PRICE_MIN, self.default_min),
            SessionStateManager.get_value(SessionStateManager.PRICE_MAX, self.default_max),
        )
