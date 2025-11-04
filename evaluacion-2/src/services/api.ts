const BASE_URL = "http://127.0.0.1:8000"; // o http://localhost:8000 si prefieres

export async function generarPregunta() {
    const res = await fetch(`${BASE_URL}/generar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
    });
    if (!res.ok) throw new Error("Error al generar pregunta");
    return await res.json();
}

export async function revisarRespuesta(pregunta: string, respuesta: string) {
    const res = await fetch(`${BASE_URL}/revisar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta, respuesta })
    });
    if (!res.ok) throw new Error("Error al revisar respuesta");
    return await res.json();
}
