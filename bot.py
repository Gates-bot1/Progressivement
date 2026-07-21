"""
TRADING BOT V1 — avec Cerveau Gemini
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fonctionnement :
  1. Lit le prix BTC/USDT sur Binance Testnet
  2. Calcule RSI + EMA
  3. Envoie tout à Gemini qui décide et explique
  4. Exécute la décision en paper trading
  5. Gemini apprend des trades passés

Auteur : Gates / Alinyxe
Version : 1.1 (avec cerveau IA)
"""

import time
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import ccxt
import numpy as np

load_dotenv()

# ── Logs ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("BOT")


# ════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════
CONFIG = {
    "symbol":            "BTC/USDT",
    "capital":           float(os.getenv("INITIAL_CAPITAL", "1000")),
    "risk_per_trade":    0.02,
    "stop_loss_pct":     1.5,
    "take_profit_pct":   3.0,
    "rsi_period":        14,
    "ema_fast":          9,
    "ema_slow":          21,
    "timeframe":         "15m",
    "candles_limit":     100,
    "cycle_seconds":     60,
    "confiance_minimum": 65,     # Gemini doit avoir >= 65% de confiance
    "paper_trading":     True,
}


# ════════════════════════════════════════════════════
# CONNEXION BINANCE TESTNET
# ════════════════════════════════════════════════════
def connect_exchange():
    try:
        exchange = ccxt.binance({
            "apiKey":          os.getenv("API_KEY", ""),
            "secret":          os.getenv("API_SECRET", ""),
            "enableRateLimit": True,
            "options":         {"defaultType": "spot"},
            "urls": {
                "api": {
                    "public":  "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }
        })
        log.info("Connecte a Binance Testnet")
        return exchange
    except Exception as e:
        log.error(f"Connexion echouee: {e}")
        return None


# ════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ════════════════════════════════════════════════════
def calcul_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag, al = gains.mean(), losses.mean()
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 2)


def calcul_ema(closes, period):
    if len(closes) < period:
        return None
    k   = 2 / (period + 1)
    ema = np.array(closes[:period]).mean()
    for p in closes[period:]:
        ema = p * k + ema * (1 - k)
    return round(float(ema), 2)


def calcul_volume_ratio(volumes, period=20):
    if len(volumes) < period + 1:
        return 1.0
    avg = np.array(volumes[-period-1:-1]).mean()
    return round(float(volumes[-1] / avg) if avg > 0 else 1.0, 2)


def calcul_tendance(closes, period=24):
    if len(closes) < period:
        return "inconnue"
    debut = closes[-period]
    fin   = closes[-1]
    pct   = (fin - debut) / debut * 100
    if pct > 2:   return f"haussiere +{pct:.1f}%"
    elif pct < -2: return f"baissiere {pct:.1f}%"
    else:          return f"laterale {pct:.1f}%"


# ════════════════════════════════════════════════════
# DONNÉES MARCHÉ
# ════════════════════════════════════════════════════
def get_market_data(exchange):
    try:
        ohlcv   = exchange.fetch_ohlcv(CONFIG["symbol"], CONFIG["timeframe"], limit=CONFIG["candles_limit"])
        closes  = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]
        return {
            "symbol":       CONFIG["symbol"],
            "prix":         closes[-1],
            "closes":       closes,
            "volumes":      volumes,
            "rsi":          calcul_rsi(closes, CONFIG["rsi_period"]),
            "ema_fast":     calcul_ema(closes, CONFIG["ema_fast"]),
            "ema_slow":     calcul_ema(closes, CONFIG["ema_slow"]),
            "volume_ratio": calcul_volume_ratio(volumes),
            "tendance":     calcul_tendance(closes),
        }
    except Exception as e:
        log.error(f"Erreur donnees marche: {e}")
        return None


# ════════════════════════════════════════════════════
# PAPER TRADER
# ════════════════════════════════════════════════════
class PaperTrader:
    def __init__(self, capital):
        self.capital        = capital
        self.capital_depart = capital
        self.trade_ouvert   = None
        self.historique     = []

    def ouvrir_trade(self, direction, prix, decision):
        if self.trade_ouvert:
            return

        taille = self.capital * CONFIG["risk_per_trade"] * 10
        sl_pct = CONFIG["stop_loss_pct"]
        tp_pct = CONFIG["take_profit_pct"]

        if direction == "LONG":
            sl = round(prix * (1 - sl_pct / 100), 2)
            tp = round(prix * (1 + tp_pct / 100), 2)
        else:
            sl = round(prix * (1 + sl_pct / 100), 2)
            tp = round(prix * (1 - tp_pct / 100), 2)

        self.trade_ouvert = {
            "direction": direction,
            "entree":    prix,
            "taille":    taille,
            "stop_loss": sl,
            "take_profit": tp,
            "heure":     datetime.now().strftime("%H:%M:%S"),
            "raison":    decision["raison"],
            "conseil":   decision.get("conseil", ""),
            "confiance": decision["confiance"],
        }

        log.info("")
        log.info(f"  ⚡ TRADE OUVERT [{direction}] — Confiance IA: {decision['confiance']}%")
        log.info(f"  Raison      : {decision['raison']}")
        if decision.get("conseil"):
            log.info(f"  Conseil IA  : {decision['conseil']}")
        log.info(f"  Prix entree : ${prix:,.2f}")
        log.info(f"  Stop-Loss   : ${sl:,.2f} (-{sl_pct}%)")
        log.info(f"  Take-Profit : ${tp:,.2f} (+{tp_pct}%)")
        log.info(f"  Taille      : ${taille:.2f}")
        log.info("")

    def verifier_sortie(self, prix, brain=None):
        if not self.trade_ouvert:
            return

        t  = self.trade_ouvert
        d  = t["direction"]
        pnl_pct = (
            (prix - t["entree"]) / t["entree"] * 100 if d == "LONG"
            else (t["entree"] - prix) / t["entree"] * 100
        )
        pnl_usd = round(pnl_pct / 100 * t["taille"], 2)

        hit_sl = (d == "LONG" and prix <= t["stop_loss"]) or \
                 (d == "SHORT" and prix >= t["stop_loss"])
        hit_tp = (d == "LONG" and prix >= t["take_profit"]) or \
                 (d == "SHORT" and prix <= t["take_profit"])

        if hit_sl or hit_tp:
            raison = "TAKE PROFIT ✅" if hit_tp else "STOP LOSS ❌"
            self.capital += pnl_usd

            resultat = {
                "direction":       d,
                "entree":          t["entree"],
                "sortie":          prix,
                "pnl_pct":         round(pnl_pct, 2),
                "pnl_usd":         pnl_usd,
                "raison_fermeture": raison,
                "raison_entree":   t["raison"],
            }
            self.historique.append(resultat)

            # Le cerveau apprend du résultat
            if brain:
                brain.enregistrer_trade(resultat)

            self.trade_ouvert = None

            log.info("")
            log.info(f"  TRADE FERME — {raison}")
            log.info(f"  P&L : {'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}% ({'+' if pnl_usd >= 0 else ''}{pnl_usd:.2f}$)")
            log.info(f"  Capital : ${self.capital:.2f}")
            log.info("")
            self._stats()

    def _stats(self):
        if not self.historique:
            return
        total  = len(self.historique)
        gains  = [t for t in self.historique if t["pnl_usd"] > 0]
        pnl    = sum(t["pnl_usd"] for t in self.historique)
        wr     = round(len(gains) / total * 100, 1)
        retour = round((self.capital - self.capital_depart) / self.capital_depart * 100, 2)
        log.info(
            f"  STATS | Trades: {total} | Win rate: {wr}% | "
            f"P&L: {'+' if pnl >= 0 else ''}{pnl:.2f}$ | "
            f"Capital: ${self.capital:.2f} ({'+' if retour >= 0 else ''}{retour}%)"
        )


# ════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ════════════════════════════════════════════════════
def main():
    log.info("=" * 55)
    log.info("  TRADING BOT V1.1 — Cerveau Gemini")
    log.info(f"  Paire    : {CONFIG['symbol']}")
    log.info(f"  Capital  : ${CONFIG['capital']:.2f}")
    log.info(f"  Confiance minimum IA : {CONFIG['confiance_minimum']}%")
    log.info(f"  Mode     : PAPER TRADING (argent fictif)")
    log.info("=" * 55)

    # Connexion exchange
    exchange = connect_exchange()
    if not exchange:
        log.error("Impossible de se connecter. Verifie tes cles API Binance.")
        return

    # Initialisation cerveau Gemini
    try:
        from brain import TradingBrain
        brain = TradingBrain()
        log.info("Cerveau Gemini pret")
    except Exception as e:
        log.error(f"Cerveau Gemini indisponible: {e}")
        log.error("Verifie GEMINI_API_KEY dans ton .env")
        return

    trader = PaperTrader(CONFIG["capital"])

    while True:
        try:
            log.info(f"--- {datetime.now().strftime('%H:%M:%S')} ---")

            # 1. Données marché
            data = get_market_data(exchange)
            if not data:
                time.sleep(30)
                continue

            log.info(
                f"Prix: ${data['prix']:,.2f} | "
                f"RSI: {data['rsi']} | "
                f"EMA9: ${data['ema_fast']:,.2f} | "
                f"EMA21: ${data['ema_slow']:,.2f} | "
                f"Volume: {data['volume_ratio']}x"
            )

            # 2. Vérifier sortie si trade ouvert
            trader.verifier_sortie(data["prix"], brain)

            # 3. Si pas de trade → demander au cerveau
            if not trader.trade_ouvert:
                log.info("Consultation du cerveau Gemini...")
                decision = brain.analyser(data)

                log.info(
                    f"Cerveau dit: {decision['action']} | "
                    f"Confiance: {decision['confiance']}% | "
                    f"{decision['raison']}"
                )

                # N'agir que si confiance suffisante
                if decision["confiance"] >= CONFIG["confiance_minimum"]:
                    if decision["action"] == "ACHAT":
                        trader.ouvrir_trade("LONG", data["prix"], decision)
                    elif decision["action"] == "VENTE":
                        trader.ouvrir_trade("SHORT", data["prix"], decision)
                else:
                    log.info(f"Confiance {decision['confiance']}% insuffisante — ATTENTE")

            else:
                t   = trader.trade_ouvert
                pnl = (data["prix"] - t["entree"]) / t["entree"] * 100
                if t["direction"] == "SHORT":
                    pnl = -pnl
                log.info(
                    f"Position [{t['direction']}] @ ${t['entree']:,.2f} | "
                    f"P&L actuel: {'+' if pnl >= 0 else ''}{pnl:.2f}%"
                )

            time.sleep(CONFIG["cycle_seconds"])

        except KeyboardInterrupt:
            log.info("Bot arrete.")
            break
        except Exception as e:
            log.error(f"Erreur: {e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
