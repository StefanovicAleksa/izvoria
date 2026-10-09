import { createContext, useContext } from "react";
import type { Lang, Me } from "./api/types";

export interface Session {
  me: Me;
  answerLanguage: Lang;
  setAnswerLanguage: (lang: Lang) => void;
  signOut: () => void;
  /** Asks the gateway again where this browser is (after signing out, or a change that affects it) */
  reload: () => void;
}

export const SessionContext = createContext<Session | null>(null);

export function useSession(): Session {
  const session = useContext(SessionContext);
  if (!session) throw new Error("useSession outside a signed-in page");
  return session;
}
