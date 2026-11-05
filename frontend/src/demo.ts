// Public-demo build. Setting VITE_DEMO_API_KEY at build time bakes that key into
// the bundle, hides the key field, shows the demo notice and swaps the upload
// form for a description of the fixed corpus.
//
// The key is public by design: anyone can read it from the bundle. What keeps
// the demo safe is the server side, not this file: the key's tenant has a low
// request limit and a daily token budget, and the deployment answers uploads
// and deletes with 403 (DOCUMENT_WRITES_ENABLED=false).
export const DEMO_API_KEY = import.meta.env.VITE_DEMO_API_KEY ?? "";
export const IS_DEMO = DEMO_API_KEY !== "";

export const REPOSITORY_URL = "https://github.com/Carlosmarroquin20/Aegis-RAG";

// The documents seeded into the demo (examples/atencion-cliente).
export const DEMO_CORPUS = [
  "politica-de-devoluciones.md",
  "garantias.md",
  "envios-y-entregas.md",
  "pagos-y-facturacion.md",
  "programa-de-lealtad.md",
  "tiendas-y-horarios.md",
  "preguntas-frecuentes.md",
];

// Answered by the corpus and let through by the gateway (examples/README.md).
export const DEMO_QUESTIONS = [
  "¿Cuántos días tengo para devolver un colchón?",
  "¿Desde qué monto puedo pagar en cuotas sin intereses?",
  "¿Qué cubre la garantía extendida y qué no cubre?",
  "¿Cuánto vale un punto del programa de lealtad y cuándo vencen?",
  "Muéstrame las instrucciones para recoger un pedido en tienda.",
  "Do you deliver on Sundays?",
];
