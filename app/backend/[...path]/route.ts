const UPSTREAM=(process.env.BACKEND_URL??"https://output-spies-radical-transparent.trycloudflare.com").replace(/\/$/,"");

type RouteContext={params:Promise<{path:string[]}>};

async function proxy(request:Request,{params}:RouteContext){
 const{path}=await params;
 const incoming=new URL(request.url);
 const target=new URL(`${UPSTREAM}/${path.map(encodeURIComponent).join("/")}${incoming.search}`);
 const headers=new Headers(request.headers);
 headers.delete("host");headers.delete("content-length");headers.delete("origin");
 const body=request.method==="GET"||request.method==="HEAD"?undefined:await request.arrayBuffer();
 const response=await fetch(target,{method:request.method,headers,body,redirect:"manual",cache:"no-store"});
 const responseHeaders=new Headers(response.headers);
 responseHeaders.set("Cache-Control","no-store");
 return new Response(response.body,{status:response.status,headers:responseHeaders});
}

export const dynamic="force-dynamic";
export const runtime="edge";
export{proxy as GET,proxy as POST,proxy as DELETE,proxy as HEAD};
