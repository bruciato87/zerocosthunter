# Come risolvere l'errore di Vercel (Senza pagare!) 🛠️

Il problema è che Vercel vede i commit provenire da `Bruciato@MacBookPro.fritz.box` e non riconosce questo utente nel tuo team.

Per risolvere, non serve pagare. Devi solo dire a Git chi sei (la tua email deve corrispondere a quella del tuo account GitHub/Vercel).

## Esegui questi comandi nel terminale:

1.  **Imposta la tua email (quella che usi su GitHub):**
    ```bash
    git config --global user.email "latuaemail@esempio.com"
    ```

2.  **Imposta il tuo nome:**
    ```bash
    git config --global user.name "Il Tuo Nome"
    ```

3.  **Correggi l'ultimo commit con la nuova identità:**
    ```bash
    git commit --amend --reset-author --no-edit
    ```

4.  **Spingi di nuovo il codice (con forza):**
    ```bash
    git push --force
    ```

Vercel vedrà il nuovo autore (Tu) e accetterà il deploy. ✅
