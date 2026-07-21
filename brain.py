"""
CERVEAU IA — Gemini
━━━━━━━━━━━━━━━━━━
Le cerveau reçoit les indicateurs du marché,
réfléchit comme un trader expérimenté,
et retourne une décision argumentée.
"""

import os
import logging
import json
import google.generativeai as genai

log = logging.getLogger("BRAIN")


class TradingBrain:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY manquant dans .env")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        # Mémoire des derniers trades pour que Gemini apprenne
        self.historique_trades = []
        log.info("Cerveau Gemini initialise")

    def analyser(self, marche: dict) -> dict:
        """
        Envoie les données marché à Gemini.
        Retourne : {action, confiance, raison, conseil}
        """
        prompt = self._construire_prompt(marche)

        try:
            response = self.model.generate_content(prompt)
            texte    = response.text.strip()
            decision = self._parser_reponse(texte)
            log.info(f"Cerveau: {decision['action']} ({decision['confiance']}%) — {decision['raison'][:80]}")
            return decision

        except Exception as e:
            log.error(f"Erreur Gemini: {e}")
            return {"action": "ATTENTE", "confiance": 0, "raison": f"Erreur IA: {e}", "conseil": ""}

    def enregistrer_trade(self, trade: dict):
        """Le cerveau mémorise le résultat du trade pour apprendre."""
        self.historique_trades.append(trade)
        if len(self.historique_trades) > 10:
            self.historique_trades.pop(0)

    def _construire_prompt(self, m: dict) -> str:
        # Résumé des derniers trades pour l'apprentissage
        contexte_trades = ""
        if self.historique_trades:
            contexte_trades = "\n\nDERNIERS TRADES (apprends de ces résultats) :"
            for t in self.historique_trades[-5:]:
                emoji = "✅" if t.get("pnl_usd", 0) > 0 else "❌"
                contexte_trades += (
                    f"\n{emoji} {t.get('direction')} | "
                    f"Entrée: ${t.get('entree')} | "
                    f"Sortie: ${t.get('sortie')} | "
                    f"P&L: {t.get('pnl_pct', 0):+.2f}% | "
                    f"Raison fermeture: {t.get('raison_fermeture', '?')}"
                )

        prompt = f"""Tu es un trader crypto professionnel expérimenté et prudent.
Analyse ces données de marché et prends une décision de trading.

DONNÉES DE MARCHÉ — {m.get('symbol', 'BTC/USDT')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prix actuel     : ${m.get('prix', 0):,.2f}
RSI (14)        : {m.get('rsi', '—')}
EMA 9           : ${m.get('ema_fast', 0):,.2f}
EMA 21          : ${m.get('ema_slow', 0):,.2f}
EMA 9 > EMA 21  : {m.get('ema_fast', 0) > m.get('ema_slow', 0)}
Volume ratio    : {m.get('volume_ratio', 1):.2f}x (vs moyenne)
Tendance 24h    : {m.get('tendance', 'inconnue')}
{contexte_trades}

RÈGLES ABSOLUES :
- Ne jamais recommander ACHAT si RSI > 70
- Ne jamais recommander VENTE si RSI < 30
- Si incertain → toujours choisir ATTENTE
- Être PRUDENT, le capital doit être protégé

Réponds UNIQUEMENT en JSON valide, sans markdown, sans explication hors JSON :
{{
  "action": "ACHAT" ou "VENTE" ou "ATTENTE",
  "confiance": nombre entre 0 et 100,
  "raison": "explication courte en français (max 100 caractères)",
  "conseil": "conseil supplémentaire court en français (max 80 caractères)"
}}"""

        return prompt

    def _parser_reponse(self, texte: str) -> dict:
        """Parse la réponse JSON de Gemini."""
        try:
            # Nettoyer si Gemini met des backticks malgré la consigne
            texte = texte.replace("```json", "").replace("```", "").strip()
            data  = json.loads(texte)

            # Validation
            action = data.get("action", "ATTENTE").upper()
            if action not in ("ACHAT", "VENTE", "ATTENTE"):
                action = "ATTENTE"

            return {
                "action":    action,
                "confiance": int(data.get("confiance", 50)),
                "raison":    str(data.get("raison", "—")),
                "conseil":   str(data.get("conseil", "")),
            }
        except Exception as e:
            log.warning(f"Parsing reponse Gemini: {e} | Reponse: {texte[:200]}")
            return {
                "action":    "ATTENTE",
                "confiance": 0,
                "raison":    "Erreur parsing reponse IA",
                "conseil":   "",
            }
