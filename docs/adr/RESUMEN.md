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
(medido: 0,974 y 0,863). Un cambio en el parser, el chunking o el modelo que empeore la búsqueda no
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
