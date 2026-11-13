# Decisiones de arquitectura — resumen en español

Resumen de los [registros de decisiones (ADR)](README.md) del proyecto. Los registros completos
están en inglés y son la versión de referencia; aquí va, para cada uno, qué se decidió, por qué
y qué cuesta, en pocas líneas.

## Arquitectura y seguridad

**[0001](0001-hexagonal-architecture-enforced-by-a-test.md) — Arquitectura hexagonal, verificada
por un test.** Cuatro capas que solo dependen hacia adentro (interfaz → aplicación → dominio ←
infraestructura). La regla no está solo documentada: un test analiza los imports de cada módulo y
falla si alguno apunta hacia afuera. Por eso cambiar ChromaDB por pgvector, u Ollama por un
proveedor alojado, es un cambio en un solo archivo.

**[0002](0002-heuristic-gateway-is-not-the-safety-boundary.md) — El gateway de entrada es una
heurística, no la frontera de seguridad.** Firmas, normalización Unicode y entropía, sin I/O y en
menos de un milisegundo. Su veredicto es una señal que encarece el ataque ingenuo; la seguridad no
depende de él, porque el prompt de sistema y el filtro de salida se aplican a todas las
peticiones. Lo que no detecta está medido y publicado.

**[0003](0003-documents-trusted-at-ingestion-untrusted-at-generation.md) — Los documentos se
aceptan al indexar y se tratan como hostiles al generar.** No se filtra el contenido al subirlo:
se defiende en el único punto donde puede hacer daño, la generación. Al indexar sí se protege al
servicio del archivo (tipo por magic bytes, límite de tamaño, presupuesto contra zip bombs).

**[0004](0004-rate-limiter-fails-open.md) — Si Redis cae, el rate limiter deja pasar.** El
limitador es un control de abuso detrás de la autenticación, no de autorización: su caída no debe
tumbar la API. Nunca es silencioso (log, métrica y alerta), y es configurable para quien prefiera
fallar cerrado.

**[0005](0005-forwarded-headers-untrusted-by-default.md) — `X-Forwarded-For` no es de confianza
salvo que se configure.** El operador declara cuántos proxies hay delante; por defecto ninguno, y
se usa la IP real de la conexión. Confiar en la cabecera sin proxy dejaría a un atacante cambiar
de IP a voluntad para evitar el bloqueo.

**[0006](0006-auth-lockout-applies-to-valid-keys.md) — El bloqueo por intentos fallidos también
rechaza claves válidas.** Si no lo hiciera, la diferencia entre 429 y 200 le diría al atacante
cuál de sus intentos acertó. Las peticiones correctas nunca cuentan para el bloqueo.

**[0007](0007-local-llm-by-default-hosted-opt-in.md) — LLM local por defecto; proveedores
alojados solo si se activan.** Por defecto ningún dato sale de la red. El modo alojado es un
único adaptador para el formato de OpenAI (sirve para Groq, OpenRouter y otros), se anuncia en los
logs al arrancar y nunca expone la clave.

**[0008](0008-output-reflection-fails-the-request.md) — Si la respuesta repite una inyección, la
petición falla.** No se "limpia" la respuesta: se devuelve un 500 genérico, se cuenta y salta una
alerta. Es 500 y no 400 porque quien falló fue el pipeline, no el usuario.

## Plataforma y operación

**[0009](0009-llm-fallback-behind-circuit-breakers.md) — Respaldo entre proveedores de LLM con
circuit breakers.** Si el proveedor principal limita o cae, responde el siguiente (normalmente
el modelo local). Un circuit breaker por backend evita pagar el timeout en cada petición durante
una caída. Medido en vivo: el respaldo mantiene el servicio, más lento y con peor calidad.

**[0010](0010-tenants-isolated-by-collection.md) — Tenants aislados por colección,
identificados por la clave, con presupuesto de tokens.** Cada equipo tiene su colección, su
límite de peticiones y su cuota diaria de tokens. El tenant sale solo de la clave, y cada ruta
llega únicamente a su colección: un filtro por metadata dependería de que ningún código lo
olvide. El archivo de tenants guarda hashes de las claves, así que no es un secreto.

**[0011](0011-immutable-supply-chain-references-and-keyless-signing.md) — Referencias
inmutables en la cadena de suministro y releases firmados sin clave.** Cada action va fijada por
SHA de commit y cada imagen base por digest, porque un tag se puede mover. Cada release publica
imágenes firmadas con cosign (identidad OIDC del workflow, sin clave que proteger), con
procedencia SLSA y SBOM.

**[0012](0012-retrieval-quality-is-a-merge-gate.md) — La calidad de recuperación es un control
del CI.** El CI mide hit@5 y MRR sobre 38 preguntas etiquetadas y falla por debajo de 0,94 y 0,82
(medido: 0,974 y 0,863). Los umbrales actuales, sobre 141 preguntas, están en 0021. Un cambio en el parser, el chunking o el modelo que empeore la búsqueda no
entra sin que se note.

**[0013](0013-one-worker-per-container.md) — Un worker por contenedor; se escala con réplicas.**
Las métricas de Prometheus viven en cada proceso: con varios workers, cada scrape vería solo una
parte del tráfico y las alertas mentirían. El estado que debe ser global (rate limit y cuotas)
vive en Redis y ya cruza réplicas.

**[0014](0014-freeze-the-startup-heap.md) — Congelar el heap del arranque.** Las pruebas de
carga mostraron pausas del garbage collector de hasta 1,3 s, que eran el p99. `gc.freeze()` al
terminar el arranque saca de su alcance los ~520.000 objetos permanentes: el p99 a 40 peticiones
por segundo bajó de 974 ms a menos de 75 ms. Una métrica y una alerta avisan si vuelve a pasar.

**[0015](0015-multilingual-embeddings-recorded-per-collection.md) — Modelo de embeddings
multilingüe, registrado en cada colección.** El modelo en inglés fallaba demasiado en español;
el multilingüe subió el hit@5 en español del 88 % al 97 %. Además, cada colección guarda qué
modelo la creó, y la API no arranca si no coincide: ChromaDB lo aceptaría en silencio y las
respuestas saldrían mal sin ningún error.

**[0016](0016-hybrid-retrieval-by-default-reranker-opt-in.md) — Búsqueda híbrida por defecto; el
reranker es opcional.** La búsqueda densa sola confunde términos exactos. Ahora se combina con BM25
(palabras clave) mediante *reciprocal rank fusion*: en las mismas 38 preguntas el MRR subió de
0,863 a 0,908 y el hit@5 de 97 % a 100 %, por unos 3 ms más por consulta. Un cross-encoder
multilingüe llega a 0,925, pero cuesta casi un segundo por consulta en CPU y 470 MB de imagen por
dos preguntas más, así que queda como opción. El índice BM25 vive en memoria, uno por colección, y
se reconstruye cuando cambia el número de fragmentos.

**[0017](0017-answer-faithfulness-judged-by-nli.md) — La fidelidad de las respuestas la juzga un
modelo NLI local, frase por frase.** Cada respuesta se divide en afirmaciones y cada una se compara
con las frases del contexto recuperado: está respaldada si alguna la implica (menos su
probabilidad de contradicción). Un LLM como juez se descartó por lento, de pago y no determinista
(0012). Sobre 80 respuestas etiquetadas detecta el 65 % de las infieles y marca por error el 7 % de
las fieles; con fragmentos enteros como premisa detectaba el 43 %. Los datos equivocados (un número
cambiado) son su punto débil: atrapa menos de la mitad. Es un control del CI y, opcionalmente, una
métrica en producción.

**[0018](0018-stream-whole-sanitized-sentences.md) — Las respuestas se transmiten como frases
completas y ya sanitizadas.** `POST /api/v1/query/stream` envía la respuesta mientras se genera
(server-sent events). El texto solo sale en frases completas, después de pasar por el
sanitizador, y con un margen de 80 caracteres revisado antes de soltar cada frase: una inyección
reflejada nunca llega al cliente, ni siquiera en parte. Todo lo que puede rechazar la consulta
corre antes del primer byte, así que un bloqueo sigue siendo 400, un presupuesto agotado 429 y
una caída del modelo 503. Si algo falla después del primer texto, un evento `error` le pide al
cliente descartar lo mostrado, y el evento final trae la respuesta completa, que es la que vale.

**[0019](0019-ml-injection-classifier-measured-not-enabled.md) — Un clasificador de inyección con
ML se mide, pero no se activa.** Sobre las mismas 326 consultas, un clasificador DeBERTa abierto
bloquea el 93 % de los ataques (el gateway, el 60 %) y cierra casi por completo sus huecos:
paráfrasis, otros idiomas, sustitución de caracteres. Pero rechaza el 17 % de las consultas
legítimas (el gateway, el 6 %), entre ellas 18 de 24 preguntas en español que solo suenan
sospechosas, con puntuaciones de 1,00 que ningún umbral separa. Queda como medición en el CI y
adaptador listo; un modo que solo registre sin bloquear sería el siguiente paso.

**[0020](0020-injection-classifier-in-shadow-mode.md) — El clasificador de inyección corre en modo
sombra, sin bloquear nunca.** Con `INJECTION_CLASSIFIER_MODE=shadow`, cada consulta que el gateway
deja pasar se puntúa en segundo plano: ninguna respuesta lo espera y ningún resultado bloquea,
retrasa ni hace fallar una consulta. Las marcas se cuentan en métricas y se registran con el hash
de la consulta (nunca el texto). El trabajo pendiente tiene límite: lo que no cabe se descarta y
se cuenta como `busy`. Así se mide, con tráfico real, lo que el corpus de 326 consultas no puede
decir; activar el bloqueo sería otra decisión, con su propio ADR.

**[0021](0021-larger-evaluation-sets.md) — Conjuntos de evaluación más grandes, y los controles
del CI recalculados sobre ellos.** La recuperación pasa de 38 a 141 preguntas (56 en inglés,
clasificadas en directas, parafraseadas y de escenario) y la fidelidad de 80 a 271 respuestas (la
mitad en inglés). Con la búsqueda híbrida, el hit@5 es 96,5 % y el MRR 0,850. Falla sobre todo en
las paráfrasis, y BM25 no ayuda nada a las preguntas en inglés sobre un corpus en español. El
reranker sube el MRR a 0,905, pero en el top 5 que recibe el modelo solo gana 2 preguntas de 141,
así que sigue siendo opcional. El juez de fidelidad es más débil de lo que mostraban 80 respuestas:
detecta el 54,6 % de las infieles y marca el 12,5 % de las fieles. 8 de sus 12 falsas alarmas sobre
respuestas correctas empiezan con un «No,» o «Yes,» suelto. Además, la mitad de las cifras
equivocadas que no detecta no aparecen en ninguna parte del documento, lo que invalida el argumento
de 0017 contra una regla de cifras. Los umbrales del CI se fijan justo debajo de lo medido. Las dos
mejoras quedan como siguientes cambios, cada una con su propia medición.

**[0022](0022-faithfulness-reply-words-and-figure-check.md) — El juez de fidelidad quita las
palabras de respuesta y comprueba las cifras contra el contexto.** Antes de juzgar se quita un
«Sí,», «No,» o «Yes,» inicial, que solo responde a la pregunta (el juez no la ve), y la frase se
sigue mostrando tal cual. Si una frase da una cifra que no aparece ni en el contexto ni en la
pregunta, se marca como no respaldada y se nombra la cifra. La misma cifra en otro formato cuenta
como la misma: «Q1.000» y «Q1,000», «8 pm» y «20:00», «dos» y «2». Sobre 271 respuestas, el juez
pasa de detectar el 54,6 % de las infieles al 67,2 %, y de marcar por error el 12,5 % de las fieles
al 7,9 %. En inglés la detección sube del 47,5 % al 65,6 %. Ahora su punto débil son las
contradicciones que reutilizan las palabras y cifras del propio documento: detecta 19 de 35.

**[0023](0023-multilingual-e5-embeddings.md) — Embeddings E5 multilingües, con sus marcas de
consulta y de pasaje.** Para mejorar las paráfrasis se compararon cinco modelos de embeddings y
cinco tamaños de fragmento sobre las 141 preguntas. El modelo por defecto pasa a ser
`multilingual-e5-small`: tiene el mismo tamaño y la misma velocidad que el anterior, y sube el MRR
de 0,850 a 0,882. E5 se entrenó con «query: » delante de la pregunta y «passage: » delante del
texto, así que la API embebe los fragmentos y las búsquedas con esas marcas. `bge-m3` en modo denso
encuentra el documento correcto entre los 5 primeros en todas las preguntas, paráfrasis incluidas
(MRR 0,957). Queda como opción documentada, porque pesa unos 1,8 GB más y tarda unos 180 ms por
consulta en CPU. Los fragmentos se quedan en 512 caracteres: ningún otro tamaño mejoró todo a la
vez. Las colecciones creadas con el modelo anterior hay que reindexarlas, o fijar el modelo
anterior en `EMBEDDING_MODEL`; la API no arranca si no coinciden.

**[0024](0024-public-demo-on-free-hosting.md) — Una demo pública: el gateway en el navegador, el
stack completo en un contenedor.** Los Spaces de Docker en Hugging Face pasaron a ser de pago, así
que la demo gratuita es una página estática en GitHub Pages que ejecuta el SecurityGateway real en
el navegador con Pyodide: el build copia exactamente los módulos que el gateway importa, un test
verifica que el árbol construido funciona solo, y un workflow la vuelve a publicar cuando cambia el
gateway. Nada de lo escrito sale de la pestaña. La imagen de un solo contenedor (nginx, API, Chroma
embebido, corpus de solo lectura, clave pública con límites) queda lista para cualquier host Docker,
pero no se publica por defecto porque en todos los revisados cuesta dinero.

**[0025](0025-mcp-server-read-only-behind-the-same-stack.md) — Un servidor MCP: una sola
herramienta de solo lectura, detrás del mismo middleware y el mismo pipeline.** Para que los
agentes (Claude Code, Cursor y otros) usen la base documental sin un envoltorio escrito a mano, la
API monta un endpoint MCP en `/mcp` (Streamable HTTP, sin estado), apagado por defecto
(`MCP_ENABLED`). Pasa por la misma pila que REST: la API key resuelve el tenant, y el límite de
peticiones, el presupuesto de tokens, el gateway y el sanitizador de salida se aplican igual.
Expone una sola herramienta, `ask`. No hay herramientas de escritura: un agente es justo el
cliente que un documento envenenado puede manipular. El resultado no incluye el fragmento crudo
de cada fuente, porque es el único texto que no pasó por el guardián de salida y lo leería el
modelo de otro. Los rechazos llegan como resultados de herramienta con error, con los mismos
mensajes vagos que en REST, y un `Origin` de navegador fuera de la lista se rechaza con 403
aunque `DEBUG` esté activo. Sirve tanto a los clientes del protocolo de 2025 como a los de la
revisión 2026-07-28, y los tests lo comprueban con el cliente oficial del SDK.

**[0026](0026-route-simple-questions-to-a-small-model.md) — Las preguntas simples van a un modelo
chico, según un clasificador de reglas.** Casi todo lo que se pregunta es una consulta puntual
(«¿Entregan los domingos?») que un modelo chico responde igual de bien, más rápido y más barato.
Con `LLM_ROUTING=heuristic`, un clasificador puro manda esas preguntas al modelo chico
(`LLM_SMALL_MODEL`, por defecto en el mismo proveedor) y deja en el grande las comparaciones,
explicaciones, cálculos, preguntas múltiples y resúmenes. Es asimétrico a propósito: basta una
señal de complejidad para quedarse en el grande, porque mandar una pregunta difícil al chico es
el error caro. Si el modelo chico falla o está limitado, responde el grande. Sobre el conjunto
etiquetado, el 82,7 % de las preguntas complejas se queda en el grande y el 97,4 % de las simples
pasa al chico; el punto débil son las preguntas que combinan dos políticas sin decirlo (1 de 8).
Mide la decisión, no las respuestas: cuánto pierde en calidad un modelo chico concreto falta
medirlo en vivo.

**[0027](0027-small-answers-in-the-wrong-language-go-to-the-large-model.md) — Una respuesta del
modelo chico en otro idioma la vuelve a generar el grande.** Medido en vivo, `gpt-oss-20b`
respondió en español 11 de 65 preguntas en inglés, porque el corpus está en español; el grande
nunca lo hizo. El idioma de una respuesta se distingue por sus palabras funcionales, sin ningún
modelo: si la pregunta y la respuesta tienen idioma y no coinciden, el router descarta la
respuesta chica y responde el grande. Si alguna de las dos no permite decidir («Q55.»), se
conserva: nunca adivina. En streaming retiene los primeros 160 caracteres hasta saber el idioma.
Los tokens descartados se cuentan aparte para que el costo no quede oculto. Sobre las respuestas
grabadas marca exactamente las 11 y ninguna del grande; las respuestas peores bajarían del 12 %
al 5,1 %, todas omisiones.
