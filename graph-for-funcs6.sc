/* graph-for-funcs.sc - Updated with Edges

   This script returns a Json representation of the graph resulting in combining the
   AST, CFG, and PDG for each method contained in the currently loaded CPG.

   Input: A valid CPG
   Output: Json with nodes and edges

   Running the Script
   ------------------
   joern --script graph-for-funcs.sc --param cpgFile=/home/kali/cpg.bin > output.json
 */

import scala.jdk.CollectionConverters._
import io.shiftleft.codepropertygraph.generated.{EdgeTypes, NodeTypes}
import io.shiftleft.codepropertygraph.generated.nodes
import io.shiftleft.semanticcpg.language._

case class NodeInfo(
  id: String,
  label: String,
  properties: scala.collection.immutable.Map[String, String],
  edges: scala.collection.immutable.List[EdgeInfo]
)

case class EdgeInfo(
  id: String,
  edgeType: String,
  in: String,
  out: String
)

case class GraphForFuncsFunction(
  function: String,
  file: String,
  id: String,
  AST: scala.collection.immutable.List[NodeInfo],
  CFG: scala.collection.immutable.List[NodeInfo],
  PDG: scala.collection.immutable.List[NodeInfo]
)

case class GraphForFuncsResult(functions: scala.collection.immutable.List[GraphForFuncsFunction])

def nodeToInfo(node: nodes.AstNode, edges: scala.collection.immutable.List[EdgeInfo]): NodeInfo = {
  val props = node.propertiesMap.asScala.map { 
    case (key, value) => (key, value.toString) 
  }.toMap
  
  NodeInfo(
    id = node.id.toString,
    label = node.label,
    properties = props,
    edges = edges
  )
}

def getEdgesForNode(node: nodes.AstNode, edgeType: String): scala.collection.immutable.List[EdgeInfo] = {
  try {
    // Get outgoing edges
    val outEdges = node._astOut.map { targetNode =>
      EdgeInfo(
        id = s"${node.id}-${edgeType}-${targetNode.id}",
        edgeType = edgeType,
        in = targetNode.id.toString,
        out = node.id.toString
      )
    }.l
    
    // Get incoming edges  
    val inEdges = node._astIn.map { sourceNode =>
      EdgeInfo(
        id = s"${sourceNode.id}-${edgeType}-${node.id}",
        edgeType = edgeType,
        in = node.id.toString,
        out = sourceNode.id.toString
      )
    }.l
    
    (outEdges ++ inEdges).distinct
  } catch {
    case _: Exception => scala.collection.immutable.List.empty[EdgeInfo]
  }
}

def getCfgEdgesForNode(node: nodes.CfgNode): scala.collection.immutable.List[EdgeInfo] = {
  try {
    // Get outgoing CFG edges
    val outEdges = node._cfgOut.map { targetNode =>
      EdgeInfo(
        id = s"${node.id}-CFG-${targetNode.id}",
        edgeType = "CFG",
        in = targetNode.id.toString,
        out = node.id.toString
      )
    }.l
    
    // Get incoming CFG edges
    val inEdges = node._cfgIn.map { sourceNode =>
      EdgeInfo(
        id = s"${sourceNode.id}-CFG-${node.id}",
        edgeType = "CFG",
        in = node.id.toString,
        out = sourceNode.id.toString
      )
    }.l
    
    (outEdges ++ inEdges).distinct
  } catch {
    case _: Exception => scala.collection.immutable.List.empty[EdgeInfo]
  }
}

def getReachingDefEdgesForNode(node: nodes.CfgNode): scala.collection.immutable.List[EdgeInfo] = {
  try {
    // Try to get reaching def edges if available
    val outEdges = node.start._reachingDefOut.map { targetNode =>
      EdgeInfo(
        id = s"${node.id}-REACHING_DEF-${targetNode.id}",
        edgeType = "REACHING_DEF",
        in = targetNode.id.toString,
        out = node.id.toString
      )
    }.l
    
    val inEdges = node.start._reachingDefIn.map { sourceNode =>
      EdgeInfo(
        id = s"${sourceNode.id}-REACHING_DEF-${node.id}",
        edgeType = "REACHING_DEF",
        in = node.id.toString,
        out = sourceNode.id.toString
      )
    }.l
    
    (outEdges ++ inEdges).distinct
  } catch {
    case _: Exception => scala.collection.immutable.List.empty[EdgeInfo]
  }
}

def generateGraphForFuncs(theCpg: io.shiftleft.codepropertygraph.Cpg): String = {
  val result = GraphForFuncsResult(
    theCpg.method.map { method =>
      val methodName = method.fullName
      val methodId = method.id.toString
      val methodFile = try {
        method.location.filename
      } catch {
        case _: Exception => "unknown"
      }

      // Get AST children with AST edges
      val astChildren = try {
        method.ast.whereNot(_.id(method.id)).l.collect {
          case node: nodes.AstNode => 
            val edges = getEdgesForNode(node, "AST")
            nodeToInfo(node, edges)
        }
      } catch {
        case _: Exception => scala.collection.immutable.List.empty[NodeInfo]
      }

      // Get CFG children with CFG edges
      val cfgChildren = try {
        method.cfgNode.whereNot(_.id(method.id)).l.collect {
          case node: nodes.CfgNode => 
            val edges = getCfgEdgesForNode(node)
            nodeToInfo(node.asInstanceOf[nodes.AstNode], edges)
        }
      } catch {
        case _: Exception => scala.collection.immutable.List.empty[NodeInfo]
      }

      // Get PDG children with data flow edges
      val pdgChildren = try {
        val locals = method.block.ast.isLocal.l
        
        if (locals.nonEmpty) {
          val sinks = locals.flatMap(_.referencingIdentifiers).dedup
          val sources = method.call.nameNot("<operator>.*").dedup
          
          if (sinks.nonEmpty && sources.nonEmpty) {
            val flows = sinks.reachableByFlows(sources).l
            
            flows.flatMap { path =>
              path.elements.collect {
                case node: nodes.CfgNode if node.id.toString != methodId => 
                  val edges = getReachingDefEdgesForNode(node)
                  nodeToInfo(node.asInstanceOf[nodes.AstNode], edges)
              }
            }.distinct
          } else {
            scala.collection.immutable.List.empty[NodeInfo]
          }
        } else {
          scala.collection.immutable.List.empty[NodeInfo]
        }
      } catch {
        case e: Exception => 
          scala.collection.immutable.List.empty[NodeInfo]
      }

      GraphForFuncsFunction(
        methodName, 
        methodFile, 
        methodId, 
        astChildren, 
        cfgChildren, 
        pdgChildren
      )
    }.l
  )

  resultToJson(result)
}

def resultToJson(result: GraphForFuncsResult): String = {
  val functionsJson = result.functions.map { func =>
    s"""{
      "function": ${escapeJson(func.function)},
      "file": ${escapeJson(func.file)},
      "id": ${escapeJson(func.id)},
      "AST": [${func.AST.map(nodeToJson).mkString(", ")}],
      "CFG": [${func.CFG.map(nodeToJson).mkString(", ")}],
      "PDG": [${func.PDG.map(nodeToJson).mkString(", ")}]
    }"""
  }.mkString(", ")
  
  s"""{"functions": [$functionsJson]}"""
}

def nodeToJson(node: NodeInfo): String = {
  val propsJson = node.properties.map { 
    case (k, v) => s"${escapeJson(k)}: ${escapeJson(v)}" 
  }.mkString(", ")
  
  val edgesJson = node.edges.map { edge =>
    s"""{
      "id": ${escapeJson(edge.id)},
      "edgeType": ${escapeJson(edge.edgeType)},
      "in": ${escapeJson(edge.in)},
      "out": ${escapeJson(edge.out)}
    }"""
  }.mkString(", ")
  
  s"""{
    "id": ${escapeJson(node.id)},
    "label": ${escapeJson(node.label)},
    "properties": {$propsJson},
    "edges": [$edgesJson]
  }"""
}

def escapeJson(s: String): String = {
  "\"" + s.replace("\\", "\\\\")
           .replace("\"", "\\\"")
           .replace("\n", "\\n")
           .replace("\r", "\\r")
           .replace("\t", "\\t") + "\""
}

// Main entry point
loadCpg("/home/kali/cpg.bin")
val jsonOutput = generateGraphForFuncs(cpg)
println(jsonOutput)
